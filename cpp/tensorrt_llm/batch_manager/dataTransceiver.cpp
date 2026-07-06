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

#include "dataTransceiver.h"

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

using BlockRange = tensorrt_llm::batch_manager::kv_cache_manager::BlockRange;

std::vector<Connection const*> const& TransferSession::getConnections() const
{
    return mConnections;
}

void TransferSession::setConnection(size_t idx, Connection const* conn)
{
    mConnections.at(idx) = conn;
}

DataContext const& TransferSession::getDataContext() const
{
    return mDataContext;
}

executor::DataTransceiverState const& TransferSession::getSelfState() const
{
    return *mSelfState;
}

executor::DataTransceiverState const& TransferSession::getOtherState() const
{
    return mOtherState;
}

runtime::BufferManager const& TransferSession::getBufferManager() const
{
    return *mBufferManager;
}

void TransferSession::send(size_t idx, void const* data, size_t size)
{
    try
    {
        mConnections.at(idx)->send(mDataContext, data, size);
    }
    catch (std::exception const& e)
    {
        throw common::RequestSpecificException(
            __FILE__, __LINE__, e.what(), mRequest->mRequestId, common::RequestErrorCode::kNETWORK_ERROR);
    }
}

void TransferSession::recv(size_t idx, void* data, size_t size)
{
    try
    {
        mConnections.at(idx)->recv(mDataContext, data, size);
    }
    catch (std::exception const& e)
    {
        throw common::RequestSpecificException(
            __FILE__, __LINE__, e.what(), mRequest->mRequestId, common::RequestErrorCode::kNETWORK_ERROR);
    }
}

LlmRequest const& TransferSession::getLlmRequest() const
{
    TLLM_CHECK(mRequest != nullptr);
    return *mRequest;
}

void TransferSession::setLlmRequest(LlmRequest const& llmRequest)
{
    for (auto const* connection : mConnections)
    {
        if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
        {
            agentConnection->activateRequestSenderState(llmRequest.mRequestId);
        }
    }
    mRequest = &llmRequest;
}

void TransferSession::eraseAgentSenderState(LlmRequest::RequestIdType requestId) const noexcept
{
    for (auto const* connection : mConnections)
    {
        try
        {
            if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
            {
                agentConnection->eraseRequestSenderState(requestId);
            }
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_WARNING("Failed to erase Agent sender state for request %ld: %s", requestId, error.what());
        }
        catch (...)
        {
            TLLM_LOG_WARNING("Failed to erase Agent sender state for request %ld", requestId);
        }
    }
}

void TransferSession::setTime(TimeNames name)
{
    if (mTimes)
    {
        mTimes->times.at(name) = LlmRequest::getSteadyClockNow();
    }
}

void TransferSession::appendMeasure(LlmRequest::TimePoint start, LlmRequest::TimePoint end, size_t size)
{
    if (mTimes)
    {
        mTimes->measures.emplace_back(Measure{start, end, size});
    }
}

void TransferSession::exportMeasure(std::ofstream& outFile, bool isContext) const
{
    if (!mTimes || mTimes->measures.empty())
    {
        return;
    }
    // write header if not exist
    if (outFile.tellp() == 0)
    {
        outFile << "RequestID,RequestInfo,Preparation,Preprocess,Transmissions,Postprocess";
        for (size_t i = 0; i < mTimes->measures.size(); i++)
        {
            outFile << ",Delay,Duration,Bandwidth(Gbps)";
        }
        outFile << '\n';
    }
    auto transferStart = mRequest->getPerfMetrics().timingMetrics.kvCacheTransferStart;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    // write measures, time is in milliseconds
    TLLM_CHECK(isContext || mRequest->getContextPhaseParams().has_value());
    auto reqId = isContext ? mRequest->mRequestId : mRequest->getContextPhaseParams().value().getReqId();
    outFile << reqId;
    auto previousTime = transferStart;
    for (auto time : mTimes->times)
    {
        if (time == LlmRequest::TimePoint())
        {
            // timepoint is unset, skip
            outFile << ",0.0";
            continue;
        }
        double delay = Milliseconds(time - previousTime).count();
        previousTime = time;
        outFile << "," << delay;
    }
    previousTime = mTimes->times[kTimePreprocess];
    for (auto const& measure : mTimes->measures)
    {
        double delay = Milliseconds(measure.start - previousTime).count();
        double duration = Milliseconds(measure.end - measure.start).count();
        double bandwidth = static_cast<double>(measure.size) * 8.0 / duration / 1e6; // byte, ms => Gbps
        outFile << "," << delay << "," << duration << "," << bandwidth;
    }
    outFile << '\n' << std::flush;
}

using runtime::SizeType32;
using AgentConnectionManager = tensorrt_llm::executor::kv_cache::AgentConnectionManager;
using DataContext = tensorrt_llm::executor::kv_cache::DataContext;

namespace
{

int32_t tagFromRequestId(LlmRequest::RequestIdType requestId)
{
    constexpr int32_t kDATA_TAG{43};
    return ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
}

DataContext chunkRequestHandshakeContext(TransferSession const& session)
{
    auto const& dataContext = session.getDataContext();
    return DataContext{dataContext.getTag() + 2, dataContext.getTransferTerminate()};
}

class AssignedRecvBufferGuard
{
public:
    AssignedRecvBufferGuard(AgentConnectionManager* manager, std::vector<std::optional<size_t>>& bufferIds)
        : mManager(manager)
        , mBufferIds(bufferIds)
    {
    }

    AssignedRecvBufferGuard(AssignedRecvBufferGuard const&) = delete;
    AssignedRecvBufferGuard& operator=(AssignedRecvBufferGuard const&) = delete;

    ~AssignedRecvBufferGuard()
    {
        if (!mActive || mManager == nullptr)
        {
            return;
        }

        auto const& managers = mManager->getCacheTransBufferManagers();
        auto const bufferCount = std::min(managers.size(), mBufferIds.size());
        for (size_t i = 0; i < bufferCount; ++i)
        {
            if (!mBufferIds[i].has_value())
            {
                continue;
            }
            auto const bufferId = mBufferIds[i].value();
            if (bufferId > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                TLLM_LOG_ERROR("Cannot release out-of-range preassigned receive-buffer index %zu", bufferId);
                continue;
            }
            try
            {
                managers[i]->freeBufferIndexForRecv(static_cast<int>(bufferId));
            }
            catch (std::exception const& error)
            {
                TLLM_LOG_ERROR("Failed to release preassigned receive-buffer index %zu: %s", bufferId, error.what());
            }
            catch (...)
            {
                TLLM_LOG_ERROR("Failed to release preassigned receive-buffer index %zu", bufferId);
            }
        }
    }

    void detach() noexcept
    {
        mActive = false;
    }

private:
    AgentConnectionManager* mManager;
    std::vector<std::optional<size_t>>& mBufferIds;
    bool mActive{true};
};

std::filesystem::path getTransferOutputPath(char const* tag)
{
    namespace fs = std::filesystem;
    auto outputPath = common::getEnvKVCacheTimeOutputPath();
    if (!outputPath.empty())
    {
        auto rank = mpi::MpiComm::world().getRank();
        auto path = fs::path(outputPath);
        fs::create_directories(path);
        return path / ("rank_" + std::to_string(rank) + "_" + tag + ".csv");
    }
    return {};
}

} // namespace

struct ReceiveCacheResource
{
    runtime::BufferManager mBufferManager;
    runtime::CudaEvent mCudaEvent;

    ReceiveCacheResource(runtime::BufferManager&& bufferManager, runtime::CudaEvent cudaEvent)
        : mBufferManager(std::move(bufferManager))
        , mCudaEvent(std::move(cudaEvent))
    {
    }
};

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState)
    : mRequestId{requestId}
    , mTransState{std::move(transState)}
{
}

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState,
    int32_t indexFromEnd, BlockKey const& lastBlockKey)
    : mRequestId{requestId}
    , mIndexFromEnd{indexFromEnd}
    , mLastBlockKey{lastBlockKey}
    , mTransState{std::move(transState)}
{
}

bool RequestInfo::operator==(RequestInfo const& rhs) const
{
    return mRequestId == rhs.mRequestId && mIndexFromEnd == rhs.mIndexFromEnd && mLastBlockKey == rhs.mLastBlockKey
        && mTransState == rhs.mTransState;
}

LlmRequest::RequestIdType RequestInfo::getRequestId() const noexcept
{
    return mRequestId;
}

executor::DataTransceiverState const& RequestInfo::getTransState() const noexcept
{
    return mTransState;
}

void RequestInfo::serialize(RequestInfo const& requestInfo, std::ostream& os)
{
    namespace su = executor::serialize_utils;
    su::serialize(requestInfo.mRequestId, os);
    su::serialize(requestInfo.mIndexFromEnd, os);
    su::serialize(requestInfo.mLastBlockKey, os);
    su::serialize(requestInfo.mTransState, os);
}

RequestInfo RequestInfo::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto requestId = su::deserialize<decltype(mRequestId)>(is);
    auto indexFromEnd = su::deserialize<decltype(mIndexFromEnd)>(is);
    auto lastBlockKey = su::deserialize<decltype(mLastBlockKey)>(is);
    auto transState = su::deserialize<decltype(mTransState)>(is);
    return RequestInfo{requestId, std::move(transState), indexFromEnd, lastBlockKey};
}

std::size_t RequestInfo::serializedSize(RequestInfo const& requestInfo)
{
    namespace su = executor::serialize_utils;
    std::size_t totalSize = 0;
    totalSize += su::serializedSize(requestInfo.mRequestId);
    totalSize += su::serializedSize(requestInfo.mIndexFromEnd);
    totalSize += su::serializedSize(requestInfo.mLastBlockKey);
    totalSize += su::serializedSize(requestInfo.mTransState);
    return totalSize;
}

class CacheSender::Impl
{
public:
    using RequestIdType = LlmRequest::RequestIdType;

    Impl(executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
        : mManager{manager}
        , mSelfState{cacheLayer.getCacheState(), executor::kv_cache::CommState{manager->getCommState()}}
        , mCacheTransferLayer{std::move(cacheLayer)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
        mCurrentRequest = std::nullopt;
        mResponseFuture = std::async(std::launch::async, &Impl::response, this);
        int asyncSendThreadNum = common::getEnvKVCacheSendMaxConcurrenceNum();
        for (int i = 0; i < asyncSendThreadNum; i++)
        {
            mAsyncSendFutures.emplace_back(
                std::async(std::launch::async, &Impl::handleAsyncSend, this, std::ref(mAsyncSendResource)));
        }
    }

    [[nodiscard]] std::future<void> sendAsync(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        TLLM_CHECK(llmRequest != nullptr);
        std::promise<void> promise;
        auto future = promise.get_future();
        llmRequest->setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
        if (mTerminate || !mManager->isRunning())
        {
            promise.set_exception(std::make_exception_ptr(TLLM_REQUEST_EXCEPTION(llmRequest->mRequestId,
                common::RequestErrorCode::kNETWORK_ERROR, "KV-cache sender service is not running")));
            return future;
        }
        {
            {
                std::scoped_lock lkResp(mSenderMutex);
                if (mTerminate)
                {
                    promise.set_exception(std::make_exception_ptr(TLLM_REQUEST_EXCEPTION(llmRequest->mRequestId,
                        common::RequestErrorCode::kNETWORK_ERROR, "KV-cache sender service is not running")));
                    return future;
                }
                mReadyResponses.emplace(llmRequest->mRequestId, Response{llmRequest, std::move(promise)});
            }
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = true;
        }
        mSenderCv.notify_all();
        return future;
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const
    {
        return mSelfState.getCommState().value();
    }

    void setCommState(executor::kv_cache::CommState commState)
    {
        mSelfState.setCommState(std::move(commState));
    }

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId)
    {
        std::unique_lock<std::mutex> lock(mMtxForMap);
        auto it = mRequestToSession.find(requestId);
        TLLM_CHECK(it != mRequestToSession.end());
        return it->second.getConnections().size();
    }

    void release(LlmRequest::RequestIdType requestId, bool exportMeasures = true)
    {
        std::unique_lock<std::mutex> lk(mMtxForMap);
        auto it = mRequestToSession.find(requestId);
        if (it == mRequestToSession.end())
        {
            mIncompatibleChunkRequestIds.erase(requestId);
            return;
        }
        it->second.eraseAgentSenderState(requestId);
        try
        {
            if (exportMeasures && !common::getEnvKVCacheTimeOutputPath().empty())
            {
                if (!mMeasuresFile.is_open())
                {
                    auto outputPath = getTransferOutputPath("send");
                    mMeasuresFile.open(outputPath);
                    TLLM_CHECK_WITH_INFO(mMeasuresFile.is_open(), "Failed to open transfer output file: %s",
                        outputPath.string().c_str());
                }
                it->second.exportMeasure(mMeasuresFile, true);
            }
        }
        catch (...)
        {
            // Session ownership must never depend on optional metrics output.
            // Erase before propagating so a failed request ID cannot retain
            // stale connections or chunk-compatibility state.
            mRequestToSession.erase(it);
            mIncompatibleChunkRequestIds.erase(requestId);
            throw;
        }
        mRequestToSession.erase(it);
        mIncompatibleChunkRequestIds.erase(requestId);
    }

    [[nodiscard]] std::optional<RequestInfo> recvRequestInfo()
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        bool isAgent = agentConnectionManager != nullptr;

        TransceiverTag::Id id;
        RequestInfo info;
        auto const* connection = isAgent
            ? agentConnectionManager->recvConnectionAndRequestInfo(info, mTerminate)
            : mManager->recvConnect(DataContext{TransceiverTag::kID_TAG, mTerminate}, &id, sizeof(id));
        if (connection == nullptr)
        {
            TLLM_LOG_WARNING("recvRequestInfo connection is nullptr, maybe the server is terminating");
            return std::nullopt;
        }

        if (!isAgent)
        {
            TLLM_CHECK(id == TransceiverTag::Id::REQUEST_SEND);
            std::uint64_t infoSize{0};
            connection->recv(DataContext{TransceiverTag::kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
            std::string serializedInfo;
            serializedInfo.resize(infoSize);
            connection->recv(DataContext{TransceiverTag::kINFO_TAG}, serializedInfo.data(), infoSize);
            std::istringstream iss(serializedInfo);
            info = RequestInfo::deserialize(iss);
        }

        auto const requestId = info.getRequestId();
        try
        {
            auto const& peerCacheState = info.getTransState().getCacheState().value();
            bool const chunkConfigMismatch = mCacheTransferLayer.getCacheState().getTransferChunkSizeBlocks()
                != peerCacheState.getTransferChunkSizeBlocks();
            if (!chunkConfigMismatch)
            {
                mCacheTransferLayer.validateSupport(info.getTransState());
            }
            else
            {
                std::unique_lock<std::mutex> lock(mMtxForMap);
                mIncompatibleChunkRequestIds.insert(requestId);
            }

            auto allCounterparts = mCacheTransferLayer.computeCounterparts(
                mSelfState.getCommState().value().getSelfIdx(), info.getTransState());

            auto peerSelfIdx = info.getTransState().getCommState()->getSelfIdx();
            int peerIdx = std::distance(
                allCounterparts.begin(), std::find(allCounterparts.begin(), allCounterparts.end(), peerSelfIdx));

            {
                TLLM_CHECK_WITH_INFO(peerIdx < static_cast<int>(allCounterparts.size()),
                    "Peer rank %d not found in expected counterparts", peerSelfIdx);
                std::unique_lock<std::mutex> lk(mMtxForMap);
                auto it = mRequestToSession.find(requestId);
                if (it == mRequestToSession.end())
                {
                    auto session = TransferSession(std::vector<Connection const*>(allCounterparts.size(), nullptr),
                        DataContext{tagFromRequestId(requestId), mTerminate}, allCounterparts, mSelfState,
                        info.getTransState(), mBufferManager, info.getIndexFromEnd(), info.getLastBlockKey(), nullptr,
                        !common::getEnvKVCacheTimeOutputPath().empty());
                    session.setTime(TransferSession::kTimeRequestInfo);
                    it = mRequestToSession.emplace(requestId, std::move(session)).first;
                }
                it->second.setConnection(peerIdx, connection);
            }
        }
        catch (...)
        {
            try
            {
                if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
                {
                    agentConnection->eraseRequestSenderState(requestId);
                }
            }
            catch (std::exception const& cleanupError)
            {
                TLLM_LOG_WARNING(
                    "Failed to erase Agent sender state for rejected request %ld: %s", requestId, cleanupError.what());
            }
            catch (...)
            {
                TLLM_LOG_WARNING("Failed to erase Agent sender state for rejected request %ld", requestId);
            }
            throw;
        }
        return info;
    }

    void sendSync(LlmRequest const& llmRequest)
    {
        TransferSession* session = nullptr;
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(llmRequest.mRequestId);
            TLLM_CHECK(it != mRequestToSession.end());
            session = std::addressof(it->second);
        }
        session->setLlmRequest(llmRequest);
        mCacheTransferLayer.format(*session);
        llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
    }

    bool cancelRequest(LlmRequest const& llmRequest)
    {
        bool isCancelled = false;
        std::scoped_lock lkResp(mSenderMutex);
        auto it = mReadyResponses.find(llmRequest.mRequestId);
        // If the request is not the current request and already in the ready queue, we can cancel it.
        if (it != mReadyResponses.end()
            && (!mCurrentRequest.has_value() || getCurrentRequestId() != llmRequest.mRequestId))
        {
            mCancelledRequests.insert(llmRequest.mRequestId);
            isCancelled = true;
        }
        else
        {
            TLLM_LOG_WARNING("Cannot cancel request %zu", llmRequest.mRequestId);
        }
        return isCancelled;
    }

    void sendReadySignal(LlmRequest::RequestIdType requestId, bool isReady)
    {
        TransferSession* session = nullptr;
        {
            std::unique_lock<std::mutex> lock(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            session = std::addressof(it->second);
        }
        auto const& connections = session->getConnections();
        // Use a request-scoped readiness channel for every same-build peer.
        // Besides avoiding cross-request signal stealing, this lets a request
        // containing inconsistently configured ranks reject all connections on
        // the same channel before deciding whether chunk phases are enabled.
        auto const readyContext = chunkRequestHandshakeContext(*session);
        for (size_t i = 0; i < connections.size(); i++)
        {
            auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
            if (agentConnectionManager)
            {
                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(i));
                TLLM_CHECK(agentConnection);
                agentConnection->sendReadySignal(readyContext, isReady);
            }
            else
            {
                connections.at(i)->send(readyContext, &isReady, sizeof(isReady));
            }
        }
    }

    bool usesChunkRequestHandshake(LlmRequest::RequestIdType requestId)
    {
        std::unique_lock<std::mutex> lock(mMtxForMap);
        auto const it = mRequestToSession.find(requestId);
        TLLM_CHECK(it != mRequestToSession.end());
        auto const& session = it->second;
        return session.getSelfState().getCacheState()->getTransferChunkSizeBlocks().has_value()
            || session.getOtherState().getCacheState()->getTransferChunkSizeBlocks().has_value();
    }

    bool hasCompatibleChunkConfig(LlmRequest::RequestIdType requestId)
    {
        std::unique_lock<std::mutex> lock(mMtxForMap);
        auto const it = mRequestToSession.find(requestId);
        TLLM_CHECK(it != mRequestToSession.end());
        auto const& session = it->second;
        return mIncompatibleChunkRequestIds.find(requestId) == mIncompatibleChunkRequestIds.end()
            && session.getSelfState().getCacheState()->getTransferChunkSizeBlocks()
            == session.getOtherState().getCacheState()->getTransferChunkSizeBlocks();
    }

    bool hasSupportedChunkTopology(LlmRequest::RequestIdType requestId)
    {
        try
        {
            std::unique_lock<std::mutex> lock(mMtxForMap);
            auto const it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            auto const& session = it->second;
            if (!session.getSelfState().getCacheState()->getTransferChunkSizeBlocks().has_value()
                && !session.getOtherState().getCacheState()->getTransferChunkSizeBlocks().has_value())
            {
                return true;
            }
            auto const& selfState = session.getSelfState();
            auto const& peerState = session.getOtherState();
            if (!cache_formatter_utils::needSendCache(selfState.getCacheState().value(),
                    peerState.getCacheState().value(), selfState.getCommState()->getSelfIdx()))
            {
                return true;
            }
            auto const pickedConnections = cache_formatter_utils::pickSendConnections(session.getConnections().size(),
                selfState.getCacheState().value(), selfState.getCommState()->getSelfIdx(),
                peerState.getCacheState().value(), session.getCounterPartRanks());
            // Every participating rank must have one peer so request- and
            // chunk-level failure decisions cannot diverge across an uneven graph.
            return pickedConnections.size() == 1;
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_WARNING(
                "Failed to validate chunked KV-cache topology for request %ld: %s", requestId, error.what());
        }
        catch (...)
        {
            TLLM_LOG_WARNING("Failed to validate chunked KV-cache topology for request %ld", requestId);
        }
        return false;
    }

    bool receiveRequesterReadySignal(LlmRequest::RequestIdType requestId)
    {
        TransferSession* session = nullptr;
        {
            std::unique_lock<std::mutex> lock(mMtxForMap);
            auto const it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            session = std::addressof(it->second);
        }

        bool allPeersReady = true;
        auto const handshakeContext = chunkRequestHandshakeContext(*session);
        for (auto const* connection : session->getConnections())
        {
            bool isReady = false;
            if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
            {
                isReady = agentConnection->recvReadySignal(handshakeContext);
            }
            else
            {
                connection->recv(handshakeContext, &isReady, sizeof(isReady));
            }
            allPeersReady &= isReady;
        }
        return allPeersReady;
    }

    void sendResponderDecisionSignal(LlmRequest::RequestIdType requestId, bool isReady)
    {
        TransferSession* session = nullptr;
        {
            std::unique_lock<std::mutex> lock(mMtxForMap);
            auto const it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            session = std::addressof(it->second);
        }

        auto const handshakeContext = chunkRequestHandshakeContext(*session);
        for (auto const* connection : session->getConnections())
        {
            if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
            {
                agentConnection->sendReadySignal(handshakeContext, isReady);
            }
            else
            {
                connection->send(handshakeContext, &isReady, sizeof(isReady));
            }
        }
    }

    bool receiveRequesterAuthorizationSignal(LlmRequest::RequestIdType requestId)
    {
        TransferSession* session = nullptr;
        {
            std::unique_lock<std::mutex> lock(mMtxForMap);
            auto const it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            session = std::addressof(it->second);
        }

        bool allAuthorized = true;
        auto const handshakeContext = chunkRequestHandshakeContext(*session);
        for (auto const* connection : session->getConnections())
        {
            bool isAuthorized = false;
            if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
            {
                isAuthorized = agentConnection->recvReadySignal(handshakeContext);
            }
            else
            {
                connection->recv(handshakeContext, &isAuthorized, sizeof(isAuthorized));
            }
            allAuthorized &= isAuthorized;
        }
        return allAuthorized;
    }

    ~Impl()
    {
        terminate();
    }

private:
    struct Response
    {
        // shared_ptr so this struct co-owns the request until the promise resolves;
        // protects worker-side dereferences and the promise itself from premature destruction.
        std::shared_ptr<LlmRequest> mRequest;
        std::promise<void> mPromise;
    };

    struct AsyncSendResource
    {
        std::deque<Response> mSendQueue;
        std::mutex mMtxForQueue;
        std::condition_variable mCVforQueue;
        std::atomic<bool> mTerminate{false};
    };

    void handleAsyncSend(AsyncSendResource& resource)
    {
        tensorrt_llm::common::setThreadName("dataTransAsyncSend");
        while (!resource.mTerminate)
        {
            Response resp;
            {
                std::unique_lock lk(resource.mMtxForQueue);
                resource.mCVforQueue.wait(
                    lk, [&resource] { return !resource.mSendQueue.empty() || resource.mTerminate; });
                if (resource.mTerminate)
                {
                    if (!resource.mSendQueue.empty())
                    {
                        TLLM_LOG_WARNING("There are still %zu requests in the mSendQueue, but encountered terminate.",
                            resource.mSendQueue.size());
                    }
                    break;
                }
                resp = std::move(resource.mSendQueue.front());
                resource.mSendQueue.pop_front();
            }
            // Sequence the read before the move: argument initializations
            // are indeterminately sequenced, so inlining resp.mRequest->...
            // alongside std::move(resp) is UB once mRequest is a shared_ptr.
            TLLM_CHECK(resp.mRequest != nullptr);
            auto const reqId = resp.mRequest->mRequestId;
            sendAndRemoveResponse(reqId, std::move(resp));
        }
    }

    void sendAndRemoveResponse(RequestIdType id, Response resp) noexcept
    {
        std::exception_ptr transferError;
        try
        {
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            sendSync(*resp.mRequest);
        }
        catch (tensorrt_llm::common::RequestSpecificException const& e)
        {
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s ", e.what());
            auto new_exception = TLLM_REQUEST_EXCEPTION(id, e.getErrorCode(), "%s", e.what());
            transferError = std::make_exception_ptr(new_exception);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s request id: %ld", e.what(), id);
            transferError = std::current_exception();
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Unknown exception in sendAndRemoveResponse for request id: %ld", id);
            transferError = std::current_exception();
        }

        try
        {
            release(id, /*exportMeasures=*/transferError == nullptr);
        }
        catch (...)
        {
            if (transferError == nullptr)
            {
                transferError = std::current_exception();
            }
            else
            {
                TLLM_LOG_WARNING("Failed to clean up KV-cache sender session for request id: %ld", id);
            }
        }

        try
        {
            if (transferError != nullptr)
            {
                resp.mPromise.set_exception(transferError);
            }
            else
            {
                resp.mPromise.set_value();
            }
        }
        catch (std::future_error const& e)
        {
            // The caller may already have abandoned or resolved the promise;
            // sender cleanup above is still complete.
            TLLM_LOG_WARNING("Failed to resolve KV-cache sender promise for request id %ld: %s", id, e.what());
        }
    }

    void asyncSendAndRemoveResponse(RequestIdType id, Response resp) noexcept
    {
        std::unique_lock lk(mAsyncSendResource.mMtxForQueue);
        mAsyncSendResource.mSendQueue.emplace_back(std::move(resp));
        mAsyncSendResource.mCVforQueue.notify_one();
    }

    void sendResponse(std::map<RequestIdType, CacheSender::Impl::Response>::iterator it)
    {
        auto reqId = mCurrentRequest.value();
        auto count = --mRemainSendCount[reqId];
        TLLM_CHECK(count >= 0);
        if (count == 0)
        {
            mRemainSendCount.erase(reqId);

            // Check if the request is cancelled
            bool isReady = true;
            {
                std::scoped_lock lk(mSenderMutex);
                if (mCancelledRequests.find(reqId) != mCancelledRequests.end())
                {
                    isReady = false;
                }
            }
            isReady &= hasCompatibleChunkConfig(reqId);
            isReady &= hasSupportedChunkTopology(reqId);
            bool const usesChunkHandshake = usesChunkRequestHandshake(reqId);
            if (usesChunkHandshake && it->second.mRequest->mSamplingConfig.beamWidth != 1)
            {
                TLLM_LOG_WARNING(
                    "Rejecting chunked KV-cache transfer for request %ld because beam width %d is unsupported.", reqId,
                    it->second.mRequest->mSamplingConfig.beamWidth);
                isReady = false;
            }
            sendReadySignal(reqId, isReady);
            bool requesterReady = true;
            bool requesterAuthorized = true;
            if (isReady)
            {
                requesterReady = receiveRequesterReadySignal(reqId);
                // Every requester that observed this responder as locally
                // ready waits for the responder's final all-requester decision.
                // This third phase makes readiness atomic across overlapping
                // multi-rank counterpart graphs.
                sendResponderDecisionSignal(reqId, requesterReady);
                // Do not enter payload transfer until every requester has
                // collected every responder decision and explicitly commits
                // the request. This prevents one ready responder from sending
                // into buffers owned by a requester that rejected another edge.
                requesterAuthorized = receiveRequesterAuthorizationSignal(reqId);
            }

            if (isReady && requesterReady && requesterAuthorized)
            {
                if (dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager) != nullptr)
                {
                    // our nixl impl seems only support recv and send in the same thread
                    //  if we use zmq as control path, we may avoid this issue
                    sendAndRemoveResponse(it->first, std::move(it->second));
                }
                else
                {
                    // if we send data in another thread, multiple rank may send data for different requests at the same
                    // time with gen DP case.
                    asyncSendAndRemoveResponse(it->first, std::move(it->second));
                }
                removeResponse(it);
            }
            else
            {
                // TODO: if the generation does not require the kv cache, the request will
                // not be removed from mCancelledRequests. This should be handled by timeout.
                auto const cancelledReqId = mCurrentRequest.value();
                Response cancelledResponse;
                {
                    std::scoped_lock lkResp(mSenderMutex);
                    auto it = mReadyResponses.find(cancelledReqId);
                    TLLM_CHECK(it != mReadyResponses.end());
                    // Move out before erasing so the promise survives the
                    // map cleanup and can be resolved (vs. destroyed unfulfilled,
                    // which would surface as std::future_error: Broken promise).
                    cancelledResponse = std::move(it->second);
                    mReadyResponses.erase(it);
                    mCancelledRequests.erase(cancelledReqId);
                    mRemainSendCount.erase(cancelledReqId);
                }
                release(cancelledReqId, /*exportMeasures=*/false);
                cancelledResponse.mPromise.set_exception(std::make_exception_ptr(TLLM_REQUEST_EXCEPTION(cancelledReqId,
                    common::RequestErrorCode::kNETWORK_ERROR,
                    "KV cache transfer for request %zu was cancelled or rejected by its requester", cancelledReqId)));
                clearCurrentRequest();
                refreshAnyReadyFromResponses();
            }
        }
        clearCurrentRequest();
    }

    void response() noexcept
    {
        try
        {
            tensorrt_llm::common::setThreadName("dataTransResp");
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            while (!mTerminate)
            {
                if (!mAnyReady)
                {
                    std::unique_lock lk(mCondMutex);
                    mSenderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                }
                if (mTerminate)
                {
                    break;
                }
                if (!hasReadyResponses())
                {
                    refreshAnyReadyFromResponses();
                    continue;
                }
                auto requestInfo = recvRequestInfo();
                if (!requestInfo.has_value() || mTerminate || !mManager->isRunning())
                {
                    if (mTerminate)
                    {
                        break;
                    }
                    TLLM_THROW("KV-cache sender service stopped while responses were pending.");
                }
                auto reqId = requestInfo->getRequestId();

                {
                    std::scoped_lock lk(mSenderMutex);
                    mCurrentRequest = reqId;
                }

                if (mRemainSendCount.find(reqId) == mRemainSendCount.end())
                {
                    mRemainSendCount[reqId] = getCounterpartsCount(reqId);
                }
                if (!hasCurrentResponse())
                {
                    std::unique_lock lk(mCondMutex);
                    mSenderCv.wait(lk, [this]() { return (hasCurrentResponse() || mTerminate); });
                    if (mTerminate)
                    {
                        break;
                    }
                }
                auto it = getCurrentResponse();
                sendResponse(it);
            }
        }
        catch (std::exception const& err)
        {
            TLLM_LOG_ERROR("Exception in CacheSender response: %s", err.what());
            failResponseService(std::current_exception());
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Unknown exception in CacheSender response");
            failResponseService(std::current_exception());
        }
    }

    void failResponseService(std::exception_ptr error) noexcept
    {
        mTerminate = true;
        std::vector<std::pair<RequestIdType, Response>> pendingResponses;
        {
            std::scoped_lock lk(mSenderMutex);
            pendingResponses.reserve(mReadyResponses.size());
            for (auto& [requestId, response] : mReadyResponses)
            {
                pendingResponses.emplace_back(requestId, std::move(response));
            }
            mReadyResponses.clear();
            mCancelledRequests.clear();
            mRemainSendCount.clear();
            mCurrentRequest = std::nullopt;
        }

        for (auto& [requestId, response] : pendingResponses)
        {
            try
            {
                release(requestId, /*exportMeasures=*/false);
            }
            catch (std::exception const& cleanupError)
            {
                TLLM_LOG_WARNING(
                    "Failed to clean up KV-cache session %ld after sender failure: %s", requestId, cleanupError.what());
            }
            catch (...)
            {
                TLLM_LOG_WARNING("Failed to clean up KV-cache session %ld after sender failure", requestId);
            }

            try
            {
                response.mPromise.set_exception(error);
            }
            catch (std::future_error const& promiseError)
            {
                TLLM_LOG_WARNING("Failed to reject KV-cache response promise %ld: %s", requestId, promiseError.what());
            }
        }
        // Agent transfers execute synchronously on this response thread, so
        // no worker can still reference a session here. Other backends may
        // have in-flight async workers and are drained by terminate().
        if (dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager) != nullptr)
        {
            clearTransferSessions();
        }

        {
            std::unique_lock lk(mCondMutex);
            mAnyReady = false;
        }
        mSenderCv.notify_all();
    }

    void terminate()
    {
        {
            std::unique_lock lk(mCondMutex);
            mTerminate = true;
        }
        // We don't have to wait for the future. If another thread is sending data, it won't pay attention
        // to the terminate flag.
        mSenderCv.notify_all();
        mAsyncSendResource.mTerminate = true;
        mAsyncSendResource.mCVforQueue.notify_all();
        for (auto& future : mAsyncSendFutures)
        {
            future.get();
        }
        if (mResponseFuture.valid())
        {
            mResponseFuture.get();
        }
        clearTransferSessions();
    }

    void clearTransferSessions() noexcept
    {
        try
        {
            std::unique_lock<std::mutex> lock(mMtxForMap);
            for (auto const& [requestId, session] : mRequestToSession)
            {
                session.eraseAgentSenderState(requestId);
            }
            mRequestToSession.clear();
            mIncompatibleChunkRequestIds.clear();
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_WARNING("Failed to clear KV-cache sender sessions: %s", error.what());
        }
        catch (...)
        {
            TLLM_LOG_WARNING("Failed to clear KV-cache sender sessions");
        }
    }

    void removeResponse(std::map<RequestIdType, Response>::iterator it)
    {
        {
            std::scoped_lock lkResp(mSenderMutex);
            mReadyResponses.erase(it);
        }
        refreshAnyReadyFromResponses();
    }

    void refreshAnyReadyFromResponses()
    {
        // The condition mutex serializes this recheck with sendAsync's true
        // transition. Locking it before the response mutex also matches the
        // condition-variable predicates below.
        std::unique_lock lkCond(mCondMutex);
        std::scoped_lock lkResp(mSenderMutex);
        mAnyReady = !mReadyResponses.empty();
    }

    [[nodiscard]] bool hasReadyResponses()
    {
        std::scoped_lock lk(mSenderMutex);
        return !mReadyResponses.empty();
    }

    [[nodiscard]] bool hasCurrentResponse()
    {
        std::scoped_lock lk(mSenderMutex);
        return mCurrentRequest.has_value() && mReadyResponses.find(*mCurrentRequest) != mReadyResponses.end();
    }

    void clearCurrentRequest()
    {
        std::scoped_lock lk(mSenderMutex);
        mCurrentRequest.reset();
    }

    [[nodiscard]] RequestIdType getCurrentRequestId() const
    {
        return mCurrentRequest.value();
    }

    [[nodiscard]] std::map<RequestIdType, Response>::iterator getCurrentResponse()
    {
        std::scoped_lock lk(mSenderMutex);
        auto it = mReadyResponses.find(getCurrentRequestId());
        TLLM_CHECK(it != mReadyResponses.end());
        return it;
    }

public:
    void setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
        std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType,
        nvinfer1::DataType ssmStateDataType)
    {
        mCacheTransferLayer.setRnnConfig(rnnModelConfig, rnnLayerNumPerPP, convStateDataType, ssmStateDataType);
        mSelfState.setCacheState(mCacheTransferLayer.getCacheState());
    }

private:
    std::optional<RequestIdType> mCurrentRequest;
    std::set<LlmRequest::RequestIdType> mCancelledRequests;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mSenderMutex, mCondMutex;
    std::atomic<bool> mAnyReady{false}, mTerminate{false};
    std::condition_variable mSenderCv, mResponderCv;
    std::future<void> mResponseFuture;
    std::unordered_map<LlmRequest::RequestIdType, int> mRemainSendCount;
    AsyncSendResource mAsyncSendResource;
    std::vector<std::future<void>> mAsyncSendFutures;
    int mDeviceId{-1};

    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, TransferSession> mRequestToSession;
    std::unordered_set<LlmRequest::RequestIdType> mIncompatibleChunkRequestIds;
    executor::DataTransceiverState mSelfState;
    CacheTransferLayer mCacheTransferLayer;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
};

class CacheReceiver::Impl
{
public:
    Impl(executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
        : mManager{manager}
        , mSelfState{cacheLayer.getCacheState(), executor::kv_cache::CommState{manager->getCommState()}}
        , mCacheTransferLayer{std::move(cacheLayer)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    }

    [[nodiscard]] std::future<void> receiveAsync(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        TLLM_CHECK(llmRequest != nullptr);
        // TODO: Modify the implementation here to avoid frequent thread creation.
        // Capture by value so the async task owns a strong reference for its lifetime.
        auto llmRequestCopy = llmRequest;
        return std::async(std::launch::async, [this, llmRequestCopy]() { requestSync(*llmRequestCopy); });
    }

    [[nodiscard]] std::future<void> requestAndReceiveAsyncMultiThreads(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        TLLM_CHECK(llmRequest != nullptr);
        try
        {
            auto promise = std::make_unique<std::promise<void>>();
            auto future = promise->get_future();
            TLLM_CHECK(llmRequest->getDataTransceiverState().getCommState().has_value());
            std::string processInfo = kDefaultProcessInfo;
            if (common::getEnvRequestKVCacheConcurrent())
            {
                processInfo = llmRequest->getDataTransceiverState().getCommState()->toString();
            }
            if (mInstanceToAsyncResource.find(processInfo) == mInstanceToAsyncResource.end())
            {

                mInstanceToAsyncResource.emplace(processInfo, std::make_unique<AsyncResource>());
                auto requestFuture = std::async(std::launch::async, &CacheReceiver::Impl::request, this,
                    std::ref(*mInstanceToAsyncResource.at(processInfo)));
                mRequestFutures.emplace_back(std::move(requestFuture));
            }
            auto& asyncResource = mInstanceToAsyncResource.at(processInfo);
            {
                std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
                asyncResource->mRequestsQueue.emplace_back(llmRequest, std::move(promise));
            }
            asyncResource->mCVforQueue.notify_all();
            return future;
        }
        catch (std::exception const& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    void receiveSync(TransferSession& session)
    {
        mCacheTransferLayer.unformat(session);
        if (!common::getEnvKVCacheTimeOutputPath().empty())
        {
            std::unique_lock<std::mutex> lock(mMeasuresFileMutex);
            if (!mMeasuresFile.is_open())
            {
                auto outputPath = getTransferOutputPath("recv");
                mMeasuresFile.open(outputPath);
                TLLM_CHECK_WITH_INFO(
                    mMeasuresFile.is_open(), "Failed to open transfer output file: %s", outputPath.string().c_str());
            }
            session.exportMeasure(mMeasuresFile, false);
        }
    }

    TransferSession sendRequestInfo(LlmRequest const& llmRequest)
    {
        uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& commState = contextState.getCommState().value();
        auto const& destCacheState = contextState.getCacheState().value();
        auto const chunkConfigMismatch = mCacheTransferLayer.getCacheState().getTransferChunkSizeBlocks()
            != destCacheState.getTransferChunkSizeBlocks();
        if (!chunkConfigMismatch)
        {
            mCacheTransferLayer.validateSupport(contextState);
        }

        RequestInfo requestInfo(requestId, mSelfState);

        if (!mCacheTransferLayer.getCacheManager()->getBlockManager().isVariableWindow())
        {
            auto* cacheManager = mCacheTransferLayer.getCacheManager();
            auto const srcPpSize = destCacheState.getParallelConfig().mPipelineParallelism;
            auto requestedBlockRange = getBlockRangeForReceiving(cacheManager, llmRequest,
                destCacheState.getEnableBlockReuse(), destCacheState.getEnablePartialReuse(),
                /*recvSideHasCP=*/false, srcPpSize);

            int32_t requestedBlockSize = requestedBlockRange.getBlockIdsPerWindow().begin()->second.size();
            // An empty Helix CP rank owns zero KV blocks for this sequence (fewer blocks than
            // cp_size). It still sends a RequestInfo so the context's per-request counterpart count
            // is satisfied, but requests zero blocks: the default RequestInfo (indexFromEnd=0, empty
            // lastBlockKey) is used and the context transmits nothing to it.
            if (requestedBlockSize > 0)
            {
                auto const beam = 0;
                auto const& uniqueTokens = llmRequest.getUniqueTokens(beam);
                auto lastBlockKey = BlockKey(
                    llmRequest.getInputTokensExtraIds().has_value(), llmRequest.getLoraTaskId(), uniqueTokens);
                auto tokensPerBlock = cacheManager->getBlockManager().getTokensPerBlock();
                SizeType32 startTokenIdx
                    = static_cast<SizeType32>(uniqueTokens.size() / tokensPerBlock) * tokensPerBlock;
                SizeType32 endTokenIdx = static_cast<SizeType32>(uniqueTokens.size());
                auto extraKeys = kv_cache_manager::generateBlockHashExtraKeys(llmRequest, startTokenIdx, endTokenIdx);
                lastBlockKey.extraKeys = std::move(extraKeys);
                int32_t indexFromEnd = requestedBlockSize - 1;

                requestInfo = RequestInfo(requestId, mSelfState, indexFromEnd, lastBlockKey);
            }
        }

        auto allCounterparts
            = mCacheTransferLayer.computeCounterparts(mSelfState.getCommState().value().getSelfIdx(), contextState);

        auto kvCounterParts = mCacheTransferLayer.getKvFormatter()->getCounterparts(
            mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx(), destCacheState);

        bool hasRnn = mCacheTransferLayer.getCacheState().hasRnnConfig() && destCacheState.hasRnnConfig();

        std::vector<SizeType32> rnnCounterParts;
        if (hasRnn)
        {
            rnnCounterParts = executor::kv_cache::targetIRanksForRnn(
                destCacheState, mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx())
                                  .mIRanks;
        }

        auto connections = mManager->getConnections(commState);
        std::vector<executor::kv_cache::Connection const*> allConnections;
        for (auto index : allCounterparts)
        {
            auto const* connection = connections.at(index);
            allConnections.emplace_back(connection);
        }

        // Finish every local allocation that can fail before publishing Agent
        // receive-buffer descriptors. Once a descriptor is sent, its slot must
        // remain quarantined until request-level completion or connection
        // teardown because a peer may still issue an RDMA write to it.
        auto const& resource = getReceiveCacheResource(llmRequest);
        auto session = TransferSession(allConnections, DataContext{tagFromRequestId(requestId), mTerminate},
            allCounterparts, mSelfState, contextState, resource->mBufferManager, requestInfo.getIndexFromEnd(),
            requestInfo.getLastBlockKey(), &llmRequest, !common::getEnvKVCacheTimeOutputPath().empty());

        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        std::vector<std::optional<size_t>> cacheBufferIds;
        AssignedRecvBufferGuard assignedBufferGuard(agentConnectionManager, cacheBufferIds);
        if (agentConnectionManager)
        {
            auto const& cacheTransBufferManagers = agentConnectionManager->getCacheTransBufferManagers();
            cacheBufferIds.reserve(cacheTransBufferManagers.size());
            std::unordered_map<uint8_t, size_t> preAssignedBufferIds;
            preAssignedBufferIds.reserve(cacheTransBufferManagers.size());
            for (auto* cacheTransBufferManager : cacheTransBufferManagers)
            {
                auto const bufferId = cacheTransBufferManager->assignBufferIndexForRecv();
                cacheBufferIds.push_back(bufferId);
                if (bufferId.has_value())
                {
                    auto const kind = static_cast<uint8_t>(cacheTransBufferManager->getBufferKind());
                    bool const inserted
                        = preAssignedBufferIds.emplace(kind, static_cast<size_t>(bufferId.value())).second;
                    TLLM_CHECK_WITH_INFO(
                        inserted, "Duplicate Agent receive-buffer kind %u", static_cast<unsigned>(kind));
                }
            }
            TLLM_CHECK(!cacheBufferIds.empty());
            session.setPreAssignedBufferIds(std::move(preAssignedBufferIds));
        }

        std::unique_lock<std::mutex> agentPublicationLock;
        if (agentConnectionManager)
        {
            agentPublicationLock = std::unique_lock<std::mutex>(mAgentPublicationMutex);
            for (auto* bufferManager : agentConnectionManager->getCacheTransBufferManagers())
            {
                TLLM_CHECK_WITH_INFO(!bufferManager->areRecvBufferAssignmentsAborted(),
                    "Agent receive-buffer publication is disabled after an earlier destination was quarantined.");
            }
        }

        bool descriptorMayBePublished = false;
        try
        {
            for (size_t ci = 0; ci < allCounterparts.size(); ci++)
            {
                auto rank = allCounterparts[ci];
                auto const* connection = connections.at(rank);

                bool isKvCounterpart
                    = std::find(kvCounterParts.begin(), kvCounterParts.end(), rank) != kvCounterParts.end();
                bool isRnnCounterpart = hasRnn
                    && std::find(rnnCounterParts.begin(), rnnCounterParts.end(), rank) != rnnCounterParts.end();

                if (agentConnectionManager)
                {
                    auto idsForRank = cacheBufferIds;
                    auto const& managers = agentConnectionManager->getCacheTransBufferManagers();
                    for (size_t i = 0; i < idsForRank.size(); i++)
                    {
                        auto kind = managers[i]->getBufferKind();
                        bool include = (kind != BufferKind::kRNN) ? isKvCounterpart : isRnnCounterpart;
                        if (!include)
                        {
                            idsForRank[i] = std::nullopt;
                        }
                    }

                    int validConnectionIdx = 0;
                    if (isKvCounterpart)
                    {
                        auto kvCpIdx
                            = std::find(kvCounterParts.begin(), kvCounterParts.end(), rank) - kvCounterParts.begin();
                        auto [pickUpIdx, localRankIdx] = mCacheTransferLayer.getKvFormatter()->pickRecvConnections(
                            allCounterparts.size(), mSelfState.getCacheState().value(),
                            mSelfState.getCommState().value().getSelfIdx(), destCacheState, allCounterparts);
                        validConnectionIdx
                            = std::find(localRankIdx.begin(), localRankIdx.end(), kvCpIdx) - localRankIdx.begin();
                    }
                    else if (isRnnCounterpart)
                    {
                        auto rnnTargetInfo = executor::kv_cache::targetIRanksForRnn(destCacheState,
                            mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx());
                        auto rnnCpIdx
                            = std::find(rnnCounterParts.begin(), rnnCounterParts.end(), rank) - rnnCounterParts.begin();
                        auto [pickUpIdx, localRankIdx]
                            = cache_formatter_utils::pickRecvConnections(rnnCounterParts.size(),
                                mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx(),
                                destCacheState, rnnCounterParts, rnnTargetInfo);
                        validConnectionIdx
                            = std::find(localRankIdx.begin(), localRankIdx.end(), rnnCpIdx) - localRankIdx.begin();
                    }

                    auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection);
                    TLLM_CHECK(agentConnection != nullptr);

                    // sendRequestAndBufferInfo may partially publish before
                    // reporting a transport error. From this point onward the
                    // slot cannot be returned to the local pool without a peer
                    // abort acknowledgement, which this transport does not expose.
                    descriptorMayBePublished = true;
                    assignedBufferGuard.detach();
                    const_cast<executor::kv_cache::AgentConnection*>(agentConnection)
                        ->sendRequestAndBufferInfo(requestInfo, idsForRank, validConnectionIdx);
                }
                else
                {
                    sendRequestInfo(connection, requestInfo);
                }
            }
        }
        catch (...)
        {
            auto const publicationError = std::current_exception();
            if (descriptorMayBePublished)
            {
                try
                {
                    abortPreAssignedRecvBufferAssignments(session);
                }
                catch (std::exception const& cleanupError)
                {
                    TLLM_LOG_WARNING("Failed to abort Agent receive-buffer assignments after publication failure: %s",
                        cleanupError.what());
                }
                catch (...)
                {
                    TLLM_LOG_WARNING("Failed to abort Agent receive-buffer assignments after publication failure.");
                }
            }
            std::rethrow_exception(publicationError);
        }
        return session;
    }

    std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest)
    {
        std::scoped_lock<std::mutex> lock(mProcessIoResouceMutex);
        TLLM_CHECK(llmRequest.getDataTransceiverState().getCommState().has_value());
        std::string processString = kDefaultProcessInfo;
        if (common::getEnvRequestKVCacheConcurrent())
        {
            processString = llmRequest.getDataTransceiverState().getCommState()->toString();
        }
        if (mProcessToResources.find(processString) == mProcessToResources.end())
        {
            mProcessToResources.emplace(processString,
                std::make_unique<ReceiveCacheResource>(
                    runtime::BufferManager{std::make_shared<runtime::CudaStream>()}, runtime::CudaEvent{}));
        }
        return mProcessToResources.at(processString);
    }

    void sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info)
    {
        std::ostringstream oss;
        RequestInfo::serialize(info, oss);
        auto const& serializedInfo = oss.str();
        std::size_t const infoSize = serializedInfo.size();
        TransceiverTag::Id id{TransceiverTag::Id::REQUEST_SEND};
        connection->send(DataContext{TransceiverTag::kID_TAG}, &id, sizeof(id));
        connection->send(DataContext{TransceiverTag::kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
        connection->send(DataContext{TransceiverTag::kINFO_TAG}, serializedInfo.data(), infoSize);
    }

    bool cancelRequest(LlmRequest const& llmRequest)
    {

        std::string processInfo = kDefaultProcessInfo;
        if (common::getEnvRequestKVCacheConcurrent())
        {
            processInfo = llmRequest.getDataTransceiverState().getCommState()->toString();
        }

        bool isCancelled = false;
        auto& asyncResource = mInstanceToAsyncResource.at(processInfo);
        {
            std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
            auto it = std::find_if(asyncResource->mRequestsQueue.begin(), asyncResource->mRequestsQueue.end(),
                [&llmRequest](RequestAndPromise const& requestAndPromise)
                { return requestAndPromise.mRequest->mRequestId == llmRequest.mRequestId; });
            if (it != asyncResource->mRequestsQueue.end())
            {
                // Resolve the promise before erasing so the future returned by
                // receiveAsync surfaces a structured cancellation error rather
                // than std::future_error: Broken promise from the destroyed promise.
                if (it->mPromise)
                {
                    try
                    {
                        it->mPromise->set_exception(std::make_exception_ptr(
                            TLLM_REQUEST_EXCEPTION(llmRequest.mRequestId, common::RequestErrorCode::kNETWORK_ERROR,
                                "Generation KV cache request cancelled before send for request %zu",
                                llmRequest.mRequestId)));
                    }
                    catch (std::future_error const&)
                    {
                        // Promise already satisfied; nothing to do.
                    }
                }
                asyncResource->mRequestsQueue.erase(it);
                isCancelled = true;
            }
            else
            {
                TLLM_LOG_WARNING("Cannot cancel request %zu", llmRequest.mRequestId);
            }
        }
        return isCancelled;
    }

    bool receiveReadySignal(TransferSession& session, std::vector<bool>* connectionReadiness = nullptr)
    {
        bool isReadyFinal = true;
        bool isReady = false;
        auto const& connections = session.getConnections();
        auto const readyContext = chunkRequestHandshakeContext(session);

        for (size_t i = 0; i < connections.size(); i++)
        {
            auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
            if (agentConnectionManager)
            {
                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(i));
                TLLM_CHECK(agentConnection);
                isReady = agentConnection->recvReadySignal(readyContext);
            }
            else
            {
                connections.at(i)->recv(readyContext, &isReady, sizeof(isReady));
            }
            isReadyFinal &= isReady;
            if (connectionReadiness != nullptr)
            {
                connectionReadiness->push_back(isReady);
            }
        }

        return isReadyFinal;
    }

    void sendAggregateReadySignal(
        TransferSession const& session, std::vector<bool> const& connectionReadiness, bool isReady)
    {
        auto const& connections = session.getConnections();
        auto const handshakeContext = chunkRequestHandshakeContext(session);
        TLLM_CHECK(connections.size() == connectionReadiness.size());
        for (size_t i = 0; i < connections.size(); ++i)
        {
            if (!connectionReadiness[i])
            {
                continue;
            }
            auto const* connection = connections[i];
            if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
            {
                agentConnection->sendReadySignal(handshakeContext, isReady);
            }
            else
            {
                connection->send(handshakeContext, &isReady, sizeof(isReady));
            }
        }
    }

    bool receiveResponderDecisionSignals(TransferSession const& session, std::vector<bool> const& connectionReadiness)
    {
        auto const& connections = session.getConnections();
        auto const handshakeContext = chunkRequestHandshakeContext(session);
        TLLM_CHECK(connections.size() == connectionReadiness.size());
        bool allPeersReady = true;
        for (size_t i = 0; i < connections.size(); ++i)
        {
            if (!connectionReadiness[i])
            {
                continue;
            }
            auto const* connection = connections[i];
            bool isReady = false;
            if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
            {
                isReady = agentConnection->recvReadySignal(handshakeContext);
            }
            else
            {
                connection->recv(handshakeContext, &isReady, sizeof(isReady));
            }
            allPeersReady &= isReady;
        }
        return allPeersReady;
    }

    void sendRequesterAuthorizationSignals(
        TransferSession const& session, std::vector<bool> const& connectionReadiness, bool isAuthorized)
    {
        auto const& connections = session.getConnections();
        auto const handshakeContext = chunkRequestHandshakeContext(session);
        TLLM_CHECK(connections.size() == connectionReadiness.size());
        for (size_t i = 0; i < connections.size(); ++i)
        {
            if (!connectionReadiness[i])
            {
                continue;
            }
            auto const* connection = connections[i];
            if (auto const* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection))
            {
                agentConnection->sendReadySignal(handshakeContext, isAuthorized);
            }
            else
            {
                connection->send(handshakeContext, &isAuthorized, sizeof(isAuthorized));
            }
        }
    }

    void releasePreAssignedRecvBuffers(TransferSession const& session)
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        if (agentConnectionManager == nullptr)
        {
            return;
        }

        auto const& managers = agentConnectionManager->getCacheTransBufferManagers();
        for (auto* bufferManager : managers)
        {
            auto const kind = static_cast<uint8_t>(bufferManager->getBufferKind());
            auto const bufferId = session.getPreAssignedBufferId(kind);
            if (bufferId.has_value())
            {
                TLLM_CHECK_WITH_INFO(bufferId.value() <= static_cast<size_t>(std::numeric_limits<int>::max()),
                    "Preassigned receive-buffer index is out of range: %zu", bufferId.value());
                bufferManager->freeBufferIndexForRecv(static_cast<int>(bufferId.value()));
            }
        }
    }

    void abortPreAssignedRecvBufferAssignments(TransferSession const& session)
    {
        if (session.getPreAssignedBufferIds().empty())
        {
            return;
        }
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        TLLM_CHECK(agentConnectionManager != nullptr);
        for (auto* bufferManager : agentConnectionManager->getCacheTransBufferManagers())
        {
            bufferManager->abortRecvBufferAssignments();
        }
    }

    void quarantinePreAssignedRecvBuffers(TransferSession const& session)
    {
        std::scoped_lock lock(mAgentPublicationMutex);
        abortPreAssignedRecvBufferAssignments(session);
    }

    ~Impl()
    {
        mTerminate.store(true);
        for (auto&& [processInfo, asyncResource] : mInstanceToAsyncResource)
        {
            asyncResource->mTerminate = true;
            asyncResource->mCVforQueue.notify_all();
        }
        for (auto&& future : mRequestFutures)
        {
            future.get();
        }
    }

private:
    void requestSync(LlmRequest& llmRequest)
    {
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "Start calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
        llmRequest.setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
        auto session = sendRequestInfo(llmRequest);
        session.setTime(TransferSession::kTimeRequestInfo);
        auto const& selfCacheState = session.getSelfState().getCacheState().value();
        auto const& peerCacheState = session.getOtherState().getCacheState().value();
        bool const usesChunkHandshake = selfCacheState.getTransferChunkSizeBlocks().has_value()
            || peerCacheState.getTransferChunkSizeBlocks().has_value();
        bool preAssignedBuffersReleased = false;
        bool payloadMayHaveStarted = false;
        bool formatterOwnsPreAssignedBuffers = false;
        auto releasePreAssignedBuffersOnce = [&]
        {
            if (!preAssignedBuffersReleased)
            {
                preAssignedBuffersReleased = true;
                releasePreAssignedRecvBuffers(session);
            }
        };
        try
        {
            std::vector<bool> connectionReadiness;
            connectionReadiness.reserve(session.getConnections().size());
            bool isReady = receiveReadySignal(session, &connectionReadiness);
            if (usesChunkHandshake)
            {
                try
                {
                    auto const pickedConnections
                        = cache_formatter_utils::pickRecvConnections(session.getConnections().size(), selfCacheState,
                            session.getSelfState().getCommState()->getSelfIdx(), peerCacheState,
                            session.getCounterPartRanks())
                              .first;
                    isReady &= pickedConnections.size() <= 1;
                }
                catch (std::exception const& error)
                {
                    TLLM_LOG_WARNING("Failed to validate chunked KV-cache receive topology for request %ld: %s",
                        llmRequest.mRequestId, error.what());
                    isReady = false;
                }
                catch (...)
                {
                    TLLM_LOG_WARNING(
                        "Failed to validate chunked KV-cache receive topology for request %ld", llmRequest.mRequestId);
                    isReady = false;
                }
            }

            // Every same-build request uses the complete readiness protocol,
            // even when this rank has chunking disabled. Otherwise per-rank
            // flag skew can let a legacy responder enter payload transfer while
            // another edge causes the requester to reject the request.
            sendAggregateReadySignal(session, connectionReadiness, isReady);
            isReady &= receiveResponderDecisionSignals(session, connectionReadiness);

            // A fourth commit phase prevents any responder from entering the
            // payload path until this requester has collected all decisions.
            // Once a positive authorization may have been delivered, Agent
            // receive slots must be quarantined on failure because a peer can
            // still hold and write the published GPU descriptor.
            if (isReady
                && std::any_of(
                    connectionReadiness.begin(), connectionReadiness.end(), [](bool ready) { return ready; }))
            {
                payloadMayHaveStarted = true;
            }
            sendRequesterAuthorizationSignals(session, connectionReadiness, isReady);
            if (!isReady)
            {
                releasePreAssignedBuffersOnce();
                // Reuse the error state for the cancelled request.
                llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
                return;
            }
            // Legacy formatters retain their existing local BufferIndexHolder
            // ownership. The chunked formatter leaves request-owned Agent slots
            // to this outer request lifecycle so failures can quarantine them.
            formatterOwnsPreAssignedBuffers = !usesChunkHandshake;
            receiveSync(session);
            if (usesChunkHandshake)
            {
                releasePreAssignedBuffersOnce();
            }
        }
        catch (...)
        {
            auto const transferError = std::current_exception();
            if (payloadMayHaveStarted && !session.getPreAssignedBufferIds().empty())
            {
                TLLM_LOG_WARNING(
                    "Quarantining Agent KV-cache receive buffers and aborting further assignments for failed request "
                    "%ld because payload transfer may have started.",
                    llmRequest.mRequestId);
                try
                {
                    quarantinePreAssignedRecvBuffers(session);
                }
                catch (std::exception const& cleanupError)
                {
                    TLLM_LOG_WARNING("Failed to abort Agent receive-buffer assignments for request %ld: %s",
                        llmRequest.mRequestId, cleanupError.what());
                }
                catch (...)
                {
                    TLLM_LOG_WARNING(
                        "Failed to abort Agent receive-buffer assignments for request %ld", llmRequest.mRequestId);
                }
            }
            else if (!formatterOwnsPreAssignedBuffers)
            {
                try
                {
                    releasePreAssignedBuffersOnce();
                }
                catch (std::exception const& cleanupError)
                {
                    TLLM_LOG_WARNING("Failed to release chunked KV-cache receive buffers for request %ld: %s",
                        llmRequest.mRequestId, cleanupError.what());
                }
                catch (...)
                {
                    TLLM_LOG_WARNING(
                        "Failed to release chunked KV-cache receive buffers for request %ld", llmRequest.mRequestId);
                }
            }
            std::rethrow_exception(transferError);
        }
        llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());

        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "End calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
    }

    struct RequestAndPromise
    {
        // shared_ptr so this struct co-owns the request until the promise resolves;
        // protects worker-side dereferences and the promise itself from premature destruction.
        std::shared_ptr<LlmRequest> mRequest;
        std::unique_ptr<std::promise<void>> mPromise;

        RequestAndPromise()
            : mRequest(nullptr)
            , mPromise(nullptr)
        {
        }

        RequestAndPromise(std::shared_ptr<LlmRequest> request, std::unique_ptr<std::promise<void>>&& promise)
            : mRequest(std::move(request))
            , mPromise(std::move(promise))
        {
        }

        RequestAndPromise(RequestAndPromise const&) = delete;

        RequestAndPromise(RequestAndPromise&& other) noexcept
            : mRequest(std::move(other.mRequest))
            , mPromise(std::move(other.mPromise))
        {
        }

        RequestAndPromise& operator=(RequestAndPromise&& other) noexcept
        {
            if (this != &other)
            {
                mRequest.reset();
                if (mPromise)
                {
                    mPromise.reset();
                }

                mRequest = std::move(other.mRequest);
                mPromise = std::move(other.mPromise);
            }
            return *this;
        }
    };

    struct AsyncResource
    {
        std::deque<RequestAndPromise> mRequestsQueue;
        std::mutex mMtxForQueue;
        std::condition_variable mCVforQueue;
        std::atomic<bool> mTerminate{false};
    };

    void request(AsyncResource& resource)
    {
        tensorrt_llm::common::setThreadName("dataTransRequest");
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

        while (!resource.mTerminate)
        {
            RequestAndPromise requestAndPromise;
            {
                std::unique_lock lck(resource.mMtxForQueue);

                resource.mCVforQueue.wait(
                    lck, [&resource] { return !resource.mRequestsQueue.empty() || resource.mTerminate; });
                if (resource.mTerminate)
                {
                    if (!resource.mRequestsQueue.empty())
                    {
                        TLLM_LOG_WARNING(
                            "There are still %zu requests in the mRequestsQueue, but encountered terminate.",
                            resource.mRequestsQueue.size());
                    }
                    break;
                }
                requestAndPromise = std::move(resource.mRequestsQueue.front());
                resource.mRequestsQueue.pop_front();
            }
            {
                try
                {
                    TLLM_CHECK_WITH_INFO(requestAndPromise.mRequest != nullptr, "requestAndPromise.mRequest is null");
                    requestSync(*requestAndPromise.mRequest);
                    requestAndPromise.mPromise->set_value();
                }
                catch (tensorrt_llm::common::RequestSpecificException const& err)
                {
                    TLLM_LOG_ERROR("Exception in DataRequester request(): request id:%zu , request context id:%zu : %s",
                        requestAndPromise.mRequest->mRequestId,
                        requestAndPromise.mRequest->getContextPhaseParams().value().getReqId(), err.what());
                    auto new_exception = TLLM_REQUEST_EXCEPTION(
                        requestAndPromise.mRequest->mRequestId, err.getErrorCode(), "%s", err.what());
                    requestAndPromise.mPromise->set_exception(std::make_exception_ptr(new_exception));
                }
                catch (std::exception const& err)
                {
                    TLLM_LOG_ERROR("Exception in CacheReceiver request(): request id:%ld , request context id:%ld : %s",
                        requestAndPromise.mRequest->mRequestId,
                        requestAndPromise.mRequest->getContextPhaseParams().value().getReqId(), err.what());
                    requestAndPromise.mPromise->set_exception(std::current_exception());
                }
            }
        }
    }

public:
    void setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
        std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType,
        nvinfer1::DataType ssmStateDataType)
    {
        mCacheTransferLayer.setRnnConfig(rnnModelConfig, rnnLayerNumPerPP, convStateDataType, ssmStateDataType);
        mSelfState.setCacheState(mCacheTransferLayer.getCacheState());
    }

private:
    int mDeviceId{-1};
    static constexpr char const* kDefaultProcessInfo = "default";
    std::vector<std::future<void>> mRequestFutures;
    std::unordered_map<std::string, std::unique_ptr<AsyncResource>> mInstanceToAsyncResource;
    executor::kv_cache::ConnectionManager* mManager;
    executor::DataTransceiverState mSelfState;
    CacheTransferLayer mCacheTransferLayer;
    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
    std::mutex mAgentPublicationMutex;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
    std::mutex mMeasuresFileMutex;
    std::atomic<bool> mTerminate{false};
};

void CacheSender::ImplDeleter::operator()(Impl* ptr)
{
    delete ptr;
}

void CacheReceiver::ImplDeleter::operator()(Impl* ptr)
{
    delete ptr;
}

CacheSender::CacheSender(
    executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
    : mImpl{std::unique_ptr<Impl, ImplDeleter>(new Impl(manager, selfIndex, std::move(cacheLayer)))}
{
}

std::future<void> CacheSender::sendAsync(std::shared_ptr<LlmRequest> const& llmRequest) const
{
    return mImpl->sendAsync(llmRequest);
}

executor::kv_cache::CommState const& CacheSender::getCommState() const
{
    return mImpl->getCommState();
}

void CacheSender::setCommState(executor::kv_cache::CommState commState)
{
    mImpl->setCommState(std::move(commState));
}

CacheSender::~CacheSender() = default;

void CacheSender::sendSync(LlmRequest const& llmRequest)
{
    mImpl->sendSync(llmRequest);
}

RequestInfo CacheSender::recvRequestInfo()
{
    auto requestInfo = mImpl->recvRequestInfo();
    TLLM_CHECK(requestInfo.has_value());
    return *requestInfo;
}

bool CacheSender::cancelRequest(LlmRequest const& llmRequest)
{
    return mImpl->cancelRequest(llmRequest);
}

void CacheSender::sendReadySignal(LlmRequest::RequestIdType requestId, bool isReady)
{
    mImpl->sendReadySignal(requestId, isReady);
}

void CacheSender::setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
    std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType, nvinfer1::DataType ssmStateDataType)
{
    mImpl->setRnnConfig(std::move(rnnModelConfig), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType);
}

CacheReceiver::CacheReceiver(
    executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
    : mImpl{std::unique_ptr<Impl, ImplDeleter>(new Impl(manager, selfIndex, std::move(cacheLayer)))}
{
}

std::future<void> CacheReceiver::receiveAsync(std::shared_ptr<LlmRequest> const& llmRequest) const
{
    return mImpl->requestAndReceiveAsyncMultiThreads(llmRequest);
}

CacheReceiver::~CacheReceiver() = default;

TransferSession CacheReceiver::sendRequestInfo(LlmRequest const& llmRequest)
{
    return mImpl->sendRequestInfo(llmRequest);
}

void CacheReceiver::receiveSync(TransferSession& session)
{
    mImpl->receiveSync(session);
}

bool CacheReceiver::cancelRequest(LlmRequest const& llmRequest)
{
    return mImpl->cancelRequest(llmRequest);
}

bool CacheReceiver::receiveReadySignal(TransferSession& session)
{
    return mImpl->receiveReadySignal(session);
}

void CacheReceiver::setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
    std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType, nvinfer1::DataType ssmStateDataType)
{
    mImpl->setRnnConfig(std::move(rnnModelConfig), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType);
}

} // namespace tensorrt_llm::batch_manager
