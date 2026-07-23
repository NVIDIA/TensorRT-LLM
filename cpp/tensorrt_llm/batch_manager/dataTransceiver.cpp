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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
    mRequest = &llmRequest;
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

void TransferSession::setReservedRecvBuffers(std::vector<BufferIndexHolder> holders)
{
    TLLM_CHECK(mReservedRecvBuffers.empty());
    mReservedRecvBuffers = std::move(holders);
}

bool TransferSession::hasReservedRecvBuffer(BaseTransBufferManager const& manager) const noexcept
{
    return std::any_of(mReservedRecvBuffers.begin(), mReservedRecvBuffers.end(),
        [&manager](BufferIndexHolder const& holder) { return holder.isBoundTo(manager); });
}

bool TransferSession::releaseReservedRecvBuffer(BaseTransBufferManager const& manager) noexcept
{
    auto const holderIt = std::find_if(mReservedRecvBuffers.begin(), mReservedRecvBuffers.end(),
        [&manager](BufferIndexHolder const& holder) { return holder.isBoundTo(manager); });
    if (holderIt == mReservedRecvBuffers.end())
    {
        return false;
    }
    holderIt->release();
    mReservedRecvBuffers.erase(holderIt);
    return true;
}

void TransferSession::releaseReservedRecvBuffers() noexcept
{
    for (auto& holder : mReservedRecvBuffers)
    {
        holder.release();
    }
    mReservedRecvBuffers.clear();
}

void TransferSession::poisonReservedRecvBuffers() noexcept
{
    for (auto& holder : mReservedRecvBuffers)
    {
        holder.poison();
    }
    mReservedRecvBuffers.clear();
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

// Bound sender-side terminal state during timeout storms. No-peer markers are
// eligible for cap-driven reclamation after the minimum retention interval,
// then move into the separately bounded finalized-handshake replay history.
constexpr std::size_t kMaxPendingPreHandshakeCancellations{65'536};
constexpr std::size_t kMaxFinalizedHandshakeIds{65'536};
constexpr auto kPreHandshakeCancellationRetention = std::chrono::seconds{30};

int32_t tagFromRequestId(LlmRequest::RequestIdType requestId)
{
    constexpr int32_t kDATA_TAG{43};
    return ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
}

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

    struct ReceivedRequestInfo
    {
        RequestInfo requestInfo;
        bool handledWithoutTransfer{false};
    };

    Impl(executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer,
        std::size_t maxPendingPreHandshakeCancellations, std::chrono::milliseconds preHandshakeCancellationRetention)
        : mManager{manager}
        , mSelfState{cacheLayer.getCacheState(), executor::kv_cache::CommState{manager->getCommState()}}
        , mCacheTransferLayer{std::move(cacheLayer)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
        , mMaxPendingPreHandshakeCancellations{maxPendingPreHandshakeCancellations}
        , mPreHandshakeCancellationRetention{preHandshakeCancellationRetention}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        TLLM_CHECK_WITH_INFO(mMaxPendingPreHandshakeCancellations > 0,
            "The maximum number of pending pre-handshake cancellations must be positive");
        TLLM_CHECK_WITH_INFO(
            mPreHandshakeCancellationRetention.count() > 0, "Pre-handshake cancellation retention must be positive");
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
        mCanPollFinalizedHandshakeReplays
            = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager) != nullptr;
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
        if (common::getEnvDisaggEnableInflightCancel())
        {
            (void) getOrCreateInFlightCancelFlag(llmRequest->mRequestId);
        }
        {
            std::scoped_lock lock(mSenderMutex);
            TLLM_CHECK_WITH_INFO(
                !mTerminate, "Cannot enqueue request %zu after CacheSender termination", llmRequest->mRequestId);
            auto const result
                = mReadyResponses.emplace(llmRequest->mRequestId, Response{llmRequest, std::move(promise)});
            TLLM_CHECK_WITH_INFO(
                result.second, "Request %zu is already queued for KV cache transfer", llmRequest->mRequestId);
        }
        mSenderCv.notify_all();
        return future;
    }

    std::shared_ptr<std::atomic<bool>> getOrCreateInFlightCancelFlag(RequestIdType requestId)
    {
        std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
        auto it = mInFlightCancelFlags.find(requestId);
        if (it != mInFlightCancelFlags.end())
        {
            return it->second;
        }
        auto flag = std::make_shared<std::atomic<bool>>(false);
        mInFlightCancelFlags.emplace(requestId, flag);
        return flag;
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const
    {
        return mSelfState.getCommState().value();
    }

    void setCommState(executor::kv_cache::CommState commState)
    {
        mSelfState.setCommState(std::move(commState));
    }

    void release(LlmRequest::RequestIdType requestId)
    {
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            if (!common::getEnvKVCacheTimeOutputPath().empty())
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
            mRequestToSession.erase(it);
        }
        if (common::getEnvDisaggEnableInflightCancel())
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            mInFlightCancelFlags.erase(requestId);
        }
    }

    void discardTransferState(LlmRequest::RequestIdType requestId) noexcept
    {
        try
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            mRequestToSession.erase(requestId);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("Failed to discard sender session for request %ld: %s", requestId, e.what());
        }
        catch (...)
        {
            TLLM_LOG_WARNING("Failed to discard sender session for request %ld: unknown exception", requestId);
        }
        if (!common::getEnvDisaggEnableInflightCancel())
        {
            return;
        }
        try
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            mInFlightCancelFlags.erase(requestId);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("Failed to discard in-flight cancel flag for request %ld: %s", requestId, e.what());
        }
        catch (...)
        {
            TLLM_LOG_WARNING("Failed to discard in-flight cancel flag for request %ld: unknown exception", requestId);
        }
    }

    [[nodiscard]] std::optional<ReceivedRequestInfo> recvRequestInfo(
        bool rejectTerminalRequest, bool waitForRequest = true)
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        bool isAgent = agentConnectionManager != nullptr;
        TLLM_CHECK_WITH_INFO(isAgent || waitForRequest, "Nonblocking RequestInfo receive requires an agent manager");

        TransceiverTag::Id id;
        RequestInfo info;
        Connection const* connection = nullptr;
        if (isAgent)
        {
            connection = waitForRequest ? agentConnectionManager->recvConnectionAndRequestInfo(info, mTerminate)
                                        : agentConnectionManager->tryRecvConnectionAndRequestInfo(info, mTerminate);
        }
        else
        {
            connection = mManager->recvConnect(DataContext{TransceiverTag::kID_TAG, mTerminate}, &id, sizeof(id));
        }
        if (connection == nullptr)
        {
            if (waitForRequest || mTerminate || !mManager->isRunning())
            {
                TLLM_LOG_WARNING("recvRequestInfo connection is nullptr, maybe the server is terminating");
            }
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

        auto requestId = info.getRequestId();
        if (rejectTerminalRequest)
        {
            bool isTerminal = false;
            {
                std::scoped_lock lock(mSenderMutex);
                isTerminal = mFinalizedHandshakeIds.find(requestId) != mFinalizedHandshakeIds.end()
                    || mPreHandshakeCancellationDeadlines.find(requestId) != mPreHandshakeCancellationDeadlines.end();
                if (isTerminal)
                {
                    armFinalizedHandshakeReplayDrainLocked();
                }
            }
            if (isTerminal)
            {
                notifyRejectedPeersNoThrow(requestId, {connection});
                return ReceivedRequestInfo{std::move(info), true};
            }
        }
        mCacheTransferLayer.validateSupport(info.getTransState());

        auto allCounterparts = mCacheTransferLayer.computeCounterparts(
            mSelfState.getCommState().value().getSelfIdx(), info.getTransState());

        auto peerSelfIdx = info.getTransState().getCommState()->getSelfIdx();
        int peerIdx = std::distance(
            allCounterparts.begin(), std::find(allCounterparts.begin(), allCounterparts.end(), peerSelfIdx));

        TLLM_CHECK_WITH_INFO(peerIdx < static_cast<int>(allCounterparts.size()),
            "Peer rank %d not found in expected counterparts", peerSelfIdx);
        std::shared_ptr<std::atomic<bool>> cancelFlag;
        if (common::getEnvDisaggEnableInflightCancel())
        {
            cancelFlag = getOrCreateInFlightCancelFlag(requestId);
        }
        bool rejectRequest = false;
        bool rejectedRequestHasSession = false;
        {
            // Publish each peer connection atomically with cancellation so a
            // queued request cannot become a transfer session between the
            // cancellation decision and session lookup.
            std::scoped_lock lock(mSenderMutex, mMtxForMap);
            if (rejectTerminalRequest
                && (mFinalizedHandshakeIds.find(requestId) != mFinalizedHandshakeIds.end()
                    || mPreHandshakeCancellationDeadlines.find(requestId) != mPreHandshakeCancellationDeadlines.end()))
            {
                // A known-terminal peer request is a late replay. Never create
                // an orphan session that would head-of-line block the response
                // worker; reject only IDs whose terminal state is explicit.
                rejectRequest = true;
                rejectedRequestHasSession = mRequestToSession.find(requestId) != mRequestToSession.end();
                armFinalizedHandshakeReplayDrainLocked();
            }
            else
            {
                auto it = mRequestToSession.find(requestId);
                if (it == mRequestToSession.end())
                {
                    auto session = cancelFlag != nullptr
                        ? TransferSession(std::vector<Connection const*>(allCounterparts.size(), nullptr),
                            DataContext{tagFromRequestId(requestId), *cancelFlag}, allCounterparts, mSelfState,
                            info.getTransState(), mBufferManager, info.getIndexFromEnd(), info.getLastBlockKey(),
                            nullptr, !common::getEnvKVCacheTimeOutputPath().empty())
                        : TransferSession(std::vector<Connection const*>(allCounterparts.size(), nullptr),
                            DataContext{tagFromRequestId(requestId), mTerminate}, allCounterparts, mSelfState,
                            info.getTransState(), mBufferManager, info.getIndexFromEnd(), info.getLastBlockKey(),
                            nullptr, !common::getEnvKVCacheTimeOutputPath().empty());
                    session.setTime(TransferSession::kTimeRequestInfo);
                    it = mRequestToSession.emplace(requestId, std::move(session)).first;
                    if (rejectTerminalRequest)
                    {
                        mRemainSendCount.emplace(requestId, allCounterparts.size());
                    }
                }
                it->second.setConnection(peerIdx, connection);
            }
        }

        if (rejectRequest)
        {
            if (!rejectedRequestHasSession && cancelFlag != nullptr)
            {
                discardTransferState(requestId);
            }
            notifyRejectedPeersNoThrow(requestId, {connection});
            return ReceivedRequestInfo{std::move(info), true};
        }
        return ReceivedRequestInfo{std::move(info), false};
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
        TLLM_LOG_DEBUG("KV cache transfer request %zu phase=transfer-submit begin.", llmRequest.mRequestId);
        mCacheTransferLayer.format(*session);
        TLLM_LOG_DEBUG("KV cache transfer request %zu phase=transfer-complete end.", llmRequest.mRequestId);
        llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
    }

    void reclaimExpiredPreHandshakeCancellationLocked()
    {
        auto const now = std::chrono::steady_clock::now();
        while (mPreHandshakeCancellationDeadlines.size() >= mMaxPendingPreHandshakeCancellations
            && !mPreHandshakeCancellationOrder.empty() && mPreHandshakeCancellationOrder.front().first <= now)
        {
            auto const [deadline, requestId] = mPreHandshakeCancellationOrder.front();
            auto const deadlineIt = mPreHandshakeCancellationDeadlines.find(requestId);
            if (deadlineIt != mPreHandshakeCancellationDeadlines.end() && deadlineIt->second == deadline)
            {
                // Move an expired no-peer marker into the bounded replay
                // history before erasing it, so a late peer remains terminal.
                recordFinalizedHandshakeLocked(requestId);
                mPreHandshakeCancellationDeadlines.erase(deadlineIt);
            }
            mPreHandshakeCancellationOrder.pop_front();
        }
    }

    bool recordCancellationLocked(RequestIdType requestId, bool isPreHandshake)
    {
        if (isPreHandshake)
        {
            if (mPreHandshakeCancellationDeadlines.find(requestId) != mPreHandshakeCancellationDeadlines.end())
            {
                return true;
            }
            if (mPreHandshakeCancellationDeadlines.size() >= mMaxPendingPreHandshakeCancellations)
            {
                reclaimExpiredPreHandshakeCancellationLocked();
                if (mPreHandshakeCancellationDeadlines.size() >= mMaxPendingPreHandshakeCancellations)
                {
                    return false;
                }
            }

            auto const deadline = std::chrono::steady_clock::now() + mPreHandshakeCancellationRetention;
            auto const [deadlineIt, inserted] = mPreHandshakeCancellationDeadlines.emplace(requestId, deadline);
            TLLM_CHECK(inserted);
            try
            {
                mPreHandshakeCancellationOrder.emplace_back(deadline, requestId);
            }
            catch (...)
            {
                mPreHandshakeCancellationDeadlines.erase(deadlineIt);
                throw;
            }
            return true;
        }

        if (mCancelledRequests.find(requestId) != mCancelledRequests.end())
        {
            return true;
        }
        mCancelledRequests.insert(requestId);
        return true;
    }

    bool cancelRequest(LlmRequest const& llmRequest)
    {
        bool const inflightCancelEnabled = common::getEnvDisaggEnableInflightCancel();
        bool isCancelled = false;
        bool isCurrentRequest = false;
        bool cancellationAdmissionDeclined = false;
        std::optional<Response> cancelledResponse;
        {
            // Serialize the sender queue and transfer-session lookup. This
            // makes pre-handshake cancellation atomic with publication of the
            // first peer RequestInfo: default-off cancellation may win before
            // any session exists, but must drain once any peer has arrived.
            std::scoped_lock lock(mSenderMutex, mMtxForMap);
            auto it = mReadyResponses.find(llmRequest.mRequestId);
            if (it != mReadyResponses.end())
            {
                isCurrentRequest = mCurrentRequest.has_value() && mCurrentRequest.value() == llmRequest.mRequestId;
                auto sessionIt = mRequestToSession.find(llmRequest.mRequestId);
                bool const hasTransferSession = sessionIt != mRequestToSession.end();
                if (!isCurrentRequest && (inflightCancelEnabled || !hasTransferSession))
                {
                    bool const isPreHandshake = !hasTransferSession;
                    if (recordCancellationLocked(llmRequest.mRequestId, isPreHandshake))
                    {
                        cancelledResponse.emplace(std::move(it->second));
                        mReadyResponses.erase(it);
                        isCancelled = true;
                    }
                    else
                    {
                        cancellationAdmissionDeclined = true;
                    }
                }
                else if (inflightCancelEnabled)
                {
                    // The legacy path cannot interrupt a current/active transfer. The opt-in path preserves the
                    // response until sendResponse coordinates ready=false or the in-flight flag stops the transfer.
                    isCancelled = recordCancellationLocked(llmRequest.mRequestId, false);
                }
            }
            else if (mCancelledRequests.find(llmRequest.mRequestId) != mCancelledRequests.end()
                || mPreHandshakeCancellationDeadlines.find(llmRequest.mRequestId)
                    != mPreHandshakeCancellationDeadlines.end())
            {
                // Cancellation is idempotent while its active or no-peer
                // terminal marker is retained.
                isCancelled = true;
            }
        }
        if (cancelledResponse.has_value())
        {
            failResponse(*cancelledResponse,
                std::make_exception_ptr(
                    TLLM_REQUEST_EXCEPTION(llmRequest.mRequestId, common::RequestErrorCode::kNETWORK_ERROR,
                        "Context KV cache request cancelled before a peer was ready for request %zu",
                        llmRequest.mRequestId)));
        }
        if (inflightCancelEnabled && !cancellationAdmissionDeclined && (!isCancelled || isCurrentRequest))
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            auto flagIt = mInFlightCancelFlags.find(llmRequest.mRequestId);
            if (flagIt != mInFlightCancelFlags.end())
            {
                flagIt->second->store(true, std::memory_order_relaxed);
                isCancelled = true;
            }
        }
        if (!isCancelled)
        {
            if (cancellationAdmissionDeclined)
            {
                TLLM_LOG_DEBUG(
                    "Cannot cancel request %zu before its peer arrives: the pending pre-handshake "
                    "cancellation limit of %zu was reached",
                    llmRequest.mRequestId, mMaxPendingPreHandshakeCancellations);
            }
            else
            {
                TLLM_LOG_WARNING("Cannot cancel request %zu", llmRequest.mRequestId);
            }
        }
        else
        {
            mSenderCv.notify_all();
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
        for (size_t i = 0; i < connections.size(); i++)
        {
            auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
            if (agentConnectionManager)
            {
                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(i));
                TLLM_CHECK(agentConnection);
                agentConnection->sendReadySignal(session->getDataContext(), isReady);
            }
            else
            {
                connections.at(i)->send(
                    executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG}, &isReady, sizeof(isReady));
            }
        }
    }

    void sendReadySignal(Connection const* connection, LlmRequest::RequestIdType requestId, bool isReady)
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        if (agentConnectionManager)
        {
            auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection);
            TLLM_CHECK(agentConnection);
            agentConnection->sendReadySignal(DataContext{tagFromRequestId(requestId), mTerminate}, isReady);
        }
        else
        {
            connection->send(
                executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG}, &isReady, sizeof(isReady));
        }
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
        try
        {
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            sendSync(*resp.mRequest);
            release(id);
            resp.mPromise.set_value();
        }
        catch (tensorrt_llm::common::RequestSpecificException const& e)
        {
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s ", e.what());
            discardTransferState(id);
            auto new_exception = TLLM_REQUEST_EXCEPTION(id, e.getErrorCode(), "%s", e.what());
            failResponse(resp, std::make_exception_ptr(new_exception));
        }
        catch (std::exception const& e)
        {
            auto const exception = std::current_exception();
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s request id: %ld", e.what(), id);
            discardTransferState(id);
            failResponse(resp, exception);
        }
        catch (...)
        {
            auto const exception = std::current_exception();
            TLLM_LOG_ERROR("Unknown exception in sendAndRemoveResponse for request id: %ld", id);
            discardTransferState(id);
            failResponse(resp, exception);
        }
    }

    void asyncSendAndRemoveResponse(RequestIdType id, Response resp) noexcept
    {
        try
        {
            std::unique_lock lock(mAsyncSendResource.mMtxForQueue);
            mAsyncSendResource.mSendQueue.emplace_back(std::move(resp));
            mAsyncSendResource.mCVforQueue.notify_one();
        }
        catch (std::exception const& err)
        {
            TLLM_LOG_ERROR("Failed to queue asynchronous KV cache send for request %zu: %s", id, err.what());
            discardTransferState(id);
            failResponse(resp, std::current_exception());
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Unknown error while queueing asynchronous KV cache send for request %zu", id);
            discardTransferState(id);
            failResponse(resp, std::current_exception());
        }
    }

    void sendResponse(RequestIdType reqId)
    {
        bool isReady = true;
        bool allCounterpartsReady = false;
        std::optional<Response> cancelledResponse;
        {
            std::scoped_lock lock(mSenderMutex);
            auto responseIt = mReadyResponses.find(reqId);
            auto countIt = mRemainSendCount.find(reqId);
            bool const isCancelled = mCancelledRequests.find(reqId) != mCancelledRequests.end();
            TLLM_CHECK(responseIt != mReadyResponses.end() || isCancelled);
            TLLM_CHECK(countIt != mRemainSendCount.end());
            auto const count = --countIt->second;
            TLLM_CHECK(count >= 0);
            if (isCancelled && responseIt != mReadyResponses.end())
            {
                cancelledResponse.emplace(std::move(responseIt->second));
                mReadyResponses.erase(responseIt);
            }
            if (count > 0)
            {
                mCurrentRequest = std::nullopt;
            }
            else
            {
                mRemainSendCount.erase(countIt);
                isReady = !isCancelled;
                allCounterpartsReady = true;
                mCurrentRequest = reqId;
            }
        }

        if (cancelledResponse.has_value())
        {
            failResponse(*cancelledResponse,
                std::make_exception_ptr(TLLM_REQUEST_EXCEPTION(reqId, common::RequestErrorCode::kNETWORK_ERROR,
                    "KV cache transfer for request %zu was cancelled", reqId)));
        }
        if (!allCounterpartsReady)
        {
            return;
        }

        // Keep mCurrentRequest set while notifying the peer so cancellation cannot change the decision after it has
        // been made. The network operation must not run under mSenderMutex.
        sendReadySignal(reqId, isReady);

        Response response;
        {
            std::scoped_lock lock(mSenderMutex);
            auto it = mReadyResponses.find(reqId);
            if (isReady)
            {
                TLLM_CHECK(it != mReadyResponses.end());
                response = std::move(it->second);
                mReadyResponses.erase(it);
            }
            recordFinalizedHandshakeLocked(reqId);
            mCancelledRequests.erase(reqId);
            mCurrentRequest = std::nullopt;
        }

        if (isReady)
        {
            if (dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager) != nullptr)
            {
                // Our NIXL implementation only supports recv and send in the same thread. Using ZMQ for the control
                // path may avoid this limitation.
                sendAndRemoveResponse(reqId, std::move(response));
                std::scoped_lock lock(mSenderMutex);
                armFinalizedHandshakeReplayDrainLocked();
            }
            else
            {
                // If we send data in another thread, multiple ranks may send data for different requests at the same
                // time with generation attention DP.
                asyncSendAndRemoveResponse(reqId, std::move(response));
            }
        }
        else
        {
            discardTransferState(reqId);
        }
    }

    void response() noexcept
    {
        std::exception_ptr responseException;
        try
        {
            tensorrt_llm::common::setThreadName("dataTransResp");
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            bool drainFinalizedReplaysImmediately = false;
            while (true)
            {
                bool pollForFinalizedReplay = false;
                {
                    std::unique_lock lock(mSenderMutex);
                    auto const hasLocalWork = [this]()
                    {
                        return mTerminate || !mReadyResponses.empty() || !mCancelledRequests.empty()
                            || !mPreHandshakeCancellationDeadlines.empty();
                    };
                    auto const canPollForFinalizedReplay = mCanPollFinalizedHandshakeReplays
                        && !mFinalizedHandshakeIds.empty()
                        && std::chrono::steady_clock::now() < mFinalizedHandshakeReplayDrainUntil;
                    if (canPollForFinalizedReplay)
                    {
                        pollForFinalizedReplay = drainFinalizedReplaysImmediately
                            || !mSenderCv.wait_for(lock, kFinalizedHandshakeReplayPollInterval, hasLocalWork);
                    }
                    else
                    {
                        mSenderCv.wait(lock, hasLocalWork);
                    }
                    if (mTerminate)
                    {
                        break;
                    }
                }

                auto receivedRequestInfo = recvRequestInfo(true, !pollForFinalizedReplay);
                if (!receivedRequestInfo.has_value())
                {
                    drainFinalizedReplaysImmediately = false;
                    if (mTerminate || !mManager->isRunning())
                    {
                        break;
                    }
                    continue;
                }
                if (receivedRequestInfo->handledWithoutTransfer)
                {
                    drainFinalizedReplaysImmediately = true;
                    continue;
                }
                drainFinalizedReplaysImmediately = false;
                auto const reqId = receivedRequestInfo->requestInfo.getRequestId();

                {
                    std::unique_lock lock(mSenderMutex);
                    mSenderCv.wait(lock,
                        [this, reqId]()
                        {
                            return mTerminate || mReadyResponses.find(reqId) != mReadyResponses.end()
                                || mCancelledRequests.find(reqId) != mCancelledRequests.end()
                                || mPreHandshakeCancellationDeadlines.find(reqId)
                                != mPreHandshakeCancellationDeadlines.end();
                        });
                    if (mTerminate)
                    {
                        mCurrentRequest = std::nullopt;
                        break;
                    }
                }
                sendResponse(reqId);
            }
        }
        catch (std::exception const& err)
        {
            TLLM_LOG_ERROR("Exception in CacheSender response: %s", err.what());
            responseException = std::current_exception();
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Unknown exception in CacheSender response");
            responseException = std::current_exception();
        }

        if (!responseException)
        {
            responseException
                = std::make_exception_ptr(std::runtime_error("CacheSender terminated before response completed"));
        }
        {
            std::scoped_lock lock(mSenderMutex);
            mTerminate = true;
        }
        mSenderCv.notify_all();
        failPendingResponses(responseException);
    }

    void terminate()
    {
        {
            std::scoped_lock lock(mSenderMutex);
            mTerminate = true;
        }
        if (common::getEnvDisaggEnableInflightCancel())
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            for (auto& [id, flag] : mInFlightCancelFlags)
            {
                flag->store(true, std::memory_order_relaxed);
            }
        }
        // Wake the sender loop and make in-flight agent transfers observe termination through
        // their per-request cancellation flags.
        mSenderCv.notify_all();
        if (mResponseFuture.valid())
        {
            mResponseFuture.get();
        }

        std::deque<Response> pendingAsyncResponses;
        {
            std::scoped_lock lock(mAsyncSendResource.mMtxForQueue);
            mAsyncSendResource.mTerminate = true;
            pendingAsyncResponses.swap(mAsyncSendResource.mSendQueue);
        }
        mAsyncSendResource.mCVforQueue.notify_all();
        for (auto& future : mAsyncSendFutures)
        {
            future.get();
        }
        auto const exception
            = std::make_exception_ptr(std::runtime_error("CacheSender terminated before asynchronous send completed"));
        for (auto& response : pendingAsyncResponses)
        {
            failResponse(response, exception);
        }
    }

    void failResponse(Response& response, std::exception_ptr const& exception) noexcept
    {
        try
        {
            response.mPromise.set_exception(exception);
        }
        catch (std::future_error const& err)
        {
            TLLM_LOG_ERROR("Failed to set CacheSender response exception: %s", err.what());
        }
    }

    void failPendingResponses(std::exception_ptr const& exception) noexcept
    {
        std::map<RequestIdType, Response> pendingResponses;
        {
            std::scoped_lock lock(mSenderMutex);
            pendingResponses.swap(mReadyResponses);
            mCurrentRequest = std::nullopt;
            mCancelledRequests.clear();
            mPreHandshakeCancellationDeadlines.clear();
            mPreHandshakeCancellationOrder.clear();
            mRemainSendCount.clear();
        }
        for (auto& entry : pendingResponses)
        {
            failResponse(entry.second, exception);
        }
    }

    void notifyRejectedPeersNoThrow(RequestIdType requestId, std::vector<Connection const*> const& connections) noexcept
    {
        TLLM_LOG_DEBUG("Rejecting a terminal KV cache handshake for request %zu", requestId);
        for (auto const* connection : connections)
        {
            try
            {
                sendReadySignal(connection, requestId, false);
            }
            catch (std::exception const& error)
            {
                TLLM_LOG_WARNING("Failed to notify a rejected peer for request %zu: %s", requestId, error.what());
            }
            catch (...)
            {
                TLLM_LOG_WARNING("Failed to notify a rejected peer for request %zu: unknown error", requestId);
            }
        }
    }

    void recordFinalizedHandshakeLocked(RequestIdType requestId)
    {
        auto const [idIt, inserted] = mFinalizedHandshakeIds.insert(requestId);
        if (inserted)
        {
            try
            {
                mFinalizedHandshakeOrder.push_back(requestId);
            }
            catch (...)
            {
                mFinalizedHandshakeIds.erase(idIt);
                throw;
            }
            if (mFinalizedHandshakeOrder.size() > kMaxFinalizedHandshakeIds)
            {
                mFinalizedHandshakeIds.erase(mFinalizedHandshakeOrder.front());
                mFinalizedHandshakeOrder.pop_front();
            }
        }
        armFinalizedHandshakeReplayDrainLocked();
    }

    void armFinalizedHandshakeReplayDrainLocked()
    {
        if (mCanPollFinalizedHandshakeReplays)
        {
            mFinalizedHandshakeReplayDrainUntil
                = std::chrono::steady_clock::now() + kFinalizedHandshakeReplayDrainWindow;
        }
    }

public:
    void setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
        std::vector<SizeType32> rnnLayerNumPerPP, tensorrt_llm::DataType convStateDataType,
        tensorrt_llm::DataType ssmStateDataType)
    {
        mCacheTransferLayer.setRnnConfig(rnnModelConfig, rnnLayerNumPerPP, convStateDataType, ssmStateDataType);
        mSelfState.setCacheState(mCacheTransferLayer.getCacheState());
    }

private:
    std::optional<RequestIdType> mCurrentRequest;
    std::set<LlmRequest::RequestIdType> mCancelledRequests;
    std::unordered_map<RequestIdType, std::chrono::steady_clock::time_point> mPreHandshakeCancellationDeadlines;
    std::deque<std::pair<std::chrono::steady_clock::time_point, RequestIdType>> mPreHandshakeCancellationOrder;
    std::deque<RequestIdType> mFinalizedHandshakeOrder;
    std::unordered_set<RequestIdType> mFinalizedHandshakeIds;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mSenderMutex;
    std::atomic<bool> mTerminate{false};
    std::condition_variable mSenderCv;
    std::future<void> mResponseFuture;
    std::unordered_map<LlmRequest::RequestIdType, int> mRemainSendCount;
    AsyncSendResource mAsyncSendResource;
    std::vector<std::future<void>> mAsyncSendFutures;
    int mDeviceId{-1};
    bool mCanPollFinalizedHandshakeReplays{false};
    std::chrono::steady_clock::time_point mFinalizedHandshakeReplayDrainUntil{};
    static constexpr auto kFinalizedHandshakeReplayPollInterval = std::chrono::milliseconds{50};
    // Cover the legacy client's short retry horizon without permanently polling the agent notification queue.
    static constexpr auto kFinalizedHandshakeReplayDrainWindow = std::chrono::seconds{30};
    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, TransferSession> mRequestToSession;
    executor::DataTransceiverState mSelfState;
    CacheTransferLayer mCacheTransferLayer;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
    std::mutex mInFlightCancelMutex;
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<std::atomic<bool>>> mInFlightCancelFlags;
    std::size_t const mMaxPendingPreHandshakeCancellations;
    std::chrono::milliseconds const mPreHandshakeCancellationRetention;
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
            std::shared_ptr<std::atomic<bool>> cancelFlag;
            if (common::getEnvDisaggEnableInflightCancel())
            {
                cancelFlag = std::make_shared<std::atomic<bool>>(false);
                std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
                mInFlightCancelFlags[llmRequest->mRequestId] = cancelFlag;
            }
            {
                std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
                asyncResource->mRequestsQueue.emplace_back(llmRequest, std::move(promise), cancelFlag);
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
        try
        {
            mCacheTransferLayer.unformat(session);
            if (!common::getEnvKVCacheTimeOutputPath().empty())
            {
                std::unique_lock<std::mutex> lock(mMeasuresFileMutex);
                if (!mMeasuresFile.is_open())
                {
                    auto outputPath = getTransferOutputPath("recv");
                    mMeasuresFile.open(outputPath);
                    TLLM_CHECK_WITH_INFO(mMeasuresFile.is_open(), "Failed to open transfer output file: %s",
                        outputPath.string().c_str());
                }
                session.exportMeasure(mMeasuresFile, false);
            }
            session.releaseReservedRecvBuffers();
        }
        catch (...)
        {
            if (common::getEnvDisaggEnableInflightCancel())
            {
                session.poisonReservedRecvBuffers();
            }
            throw;
        }
    }

    TransferSession sendRequestInfo(LlmRequest const& llmRequest, std::atomic<bool> const* perRequestCancel = nullptr)
    {
        uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& commState = contextState.getCommState().value();
        auto const& destCacheState = contextState.getCacheState().value();
        mCacheTransferLayer.validateSupport(contextState);

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

        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        std::vector<BufferIndexHolder> recvHolders;
        std::vector<std::optional<size_t>> cacheBufferIds;
        if (agentConnectionManager)
        {
            auto const* bufferCancel = common::getEnvDisaggEnableInflightCancel() ? perRequestCancel : nullptr;
            auto const& managers = agentConnectionManager->getCacheTransBufferManagers();
            recvHolders.reserve(managers.size());
            cacheBufferIds.reserve(managers.size());
            for (auto& cacheTransBufferManager : managers)
            {
                auto rawIdx = cacheTransBufferManager->assignBufferIndexForRecv(bufferCancel);
                recvHolders.emplace_back(*cacheTransBufferManager, rawIdx, /*isRecv=*/true);
                if (rawIdx.has_value())
                {
                    cacheBufferIds.push_back(static_cast<size_t>(rawIdx.value()));
                }
                else
                {
                    cacheBufferIds.push_back(std::nullopt);
                }
            }
            TLLM_CHECK(!cacheBufferIds.empty());
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

        if (common::getEnvDisaggEnableInflightCancel() && perRequestCancel != nullptr
            && perRequestCancel->load(std::memory_order_relaxed))
        {
            TLLM_THROW("KV cache receive request cancelled before publishing receive buffers");
        }

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

                    const_cast<executor::kv_cache::AgentConnection*>(agentConnection)
                        ->sendRequestAndBufferInfo(requestInfo, idsForRank, validConnectionIdx, perRequestCancel);
                }
                else
                {
                    sendRequestInfo(connection, requestInfo);
                }
            }

            auto const& resource = getReceiveCacheResource(llmRequest);
            TransferSession session = perRequestCancel != nullptr
                ? TransferSession(std::move(allConnections),
                    DataContext{tagFromRequestId(requestId), *perRequestCancel}, std::move(allCounterparts), mSelfState,
                    contextState, resource->mBufferManager, requestInfo.getIndexFromEnd(),
                    requestInfo.getLastBlockKey(), &llmRequest, !common::getEnvKVCacheTimeOutputPath().empty())
                : TransferSession(std::move(allConnections), DataContext{tagFromRequestId(requestId), mTerminate},
                    std::move(allCounterparts), mSelfState, contextState, resource->mBufferManager,
                    requestInfo.getIndexFromEnd(), requestInfo.getLastBlockKey(), &llmRequest,
                    !common::getEnvKVCacheTimeOutputPath().empty());
            if (!recvHolders.empty())
            {
                session.setReservedRecvBuffers(std::move(recvHolders));
            }
            return session;
        }
        catch (...)
        {
            if (common::getEnvDisaggEnableInflightCancel())
            {
                for (auto& holder : recvHolders)
                {
                    holder.poison();
                }
            }
            throw;
        }
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
        if (!common::getEnvDisaggEnableInflightCancel())
        {
            TLLM_LOG_WARNING(
                "Cannot cancel generation request %zu while in-flight cancellation is disabled", llmRequest.mRequestId);
            return false;
        }

        std::string processInfo = kDefaultProcessInfo;
        if (common::getEnvRequestKVCacheConcurrent())
        {
            auto const& commState = llmRequest.getDataTransceiverState().getCommState();
            if (!commState.has_value())
            {
                TLLM_LOG_WARNING("Cannot cancel request %zu: the request has no data-transceiver communication state",
                    llmRequest.mRequestId);
                return false;
            }
            processInfo = commState->toString();
        }

        auto const resourceIt = mInstanceToAsyncResource.find(processInfo);
        if (resourceIt == mInstanceToAsyncResource.end())
        {
            TLLM_LOG_WARNING("Cannot cancel request %zu: receive worker %s is not registered", llmRequest.mRequestId,
                processInfo.c_str());
            return false;
        }

        bool isCancelled = false;
        auto& asyncResource = resourceIt->second;
        std::optional<LlmRequest::RequestIdType> queuedCancelledReqId;
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
                queuedCancelledReqId = llmRequest.mRequestId;
            }
        }
        if (common::getEnvDisaggEnableInflightCancel() && queuedCancelledReqId.has_value())
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            mInFlightCancelFlags.erase(*queuedCancelledReqId);
        }
        if (!isCancelled && common::getEnvDisaggEnableInflightCancel())
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            auto flagIt = mInFlightCancelFlags.find(llmRequest.mRequestId);
            if (flagIt != mInFlightCancelFlags.end())
            {
                flagIt->second->store(true, std::memory_order_relaxed);
                isCancelled = true;
            }
        }
        if (!isCancelled)
        {
            TLLM_LOG_WARNING("Cannot cancel request %zu", llmRequest.mRequestId);
        }
        return isCancelled;
    }

    enum class ReadySignalResult
    {
        kReady,
        kNotReady,
        kMixed,
        kCancelled,
    };

    ReadySignalResult receiveReadySignalDetailed(TransferSession& session, std::atomic<bool> const& perRequestCancel)
    {
        bool isReady = false;
        bool anyReady = false;
        bool anyNotReady = false;
        auto const& connections = session.getConnections();
        for (size_t i = 0; i < connections.size(); i++)
        {
            if (perRequestCancel.load(std::memory_order_relaxed))
            {
                return ReadySignalResult::kCancelled;
            }
            auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
            if (agentConnectionManager)
            {
                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(i));
                TLLM_CHECK(agentConnection);
                auto ready = agentConnection->recvReadySignalWithStatus(
                    executor::kv_cache::DataContext{session.getDataContext().getTag(), perRequestCancel});
                if (!ready.has_value())
                {
                    return ReadySignalResult::kCancelled;
                }
                isReady = ready.value();
            }
            else
            {
                connections.at(i)->recv(
                    executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG}, &isReady, sizeof(isReady));
                if (perRequestCancel.load(std::memory_order_relaxed))
                {
                    return ReadySignalResult::kCancelled;
                }
            }
            anyReady |= isReady;
            anyNotReady |= !isReady;
        }

        if (anyReady && anyNotReady)
        {
            return ReadySignalResult::kMixed;
        }
        return anyReady ? ReadySignalResult::kReady : ReadySignalResult::kNotReady;
    }

    bool receiveReadySignal(TransferSession& session)
    {
        auto const result = receiveReadySignalDetailed(session, mTerminate);
        if (result == ReadySignalResult::kNotReady)
        {
            session.releaseReservedRecvBuffers();
        }
        else if (result == ReadySignalResult::kMixed)
        {
            if (common::getEnvDisaggEnableInflightCancel())
            {
                session.poisonReservedRecvBuffers();
            }
            else
            {
                session.releaseReservedRecvBuffers();
            }
        }
        return result == ReadySignalResult::kReady;
    }

    ~Impl()
    {
        mTerminate.store(true);
        if (common::getEnvDisaggEnableInflightCancel())
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            for (auto& [id, flag] : mInFlightCancelFlags)
            {
                flag->store(true, std::memory_order_relaxed);
            }
        }
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
    void requestSync(LlmRequest& llmRequest, std::atomic<bool> const& perRequestCancel)
    {
        auto const requestId = llmRequest.mRequestId;
        auto const contextRequestId = llmRequest.getContextPhaseParams().value().getReqId();
        char const* phase = "request-info";
        TLLM_LOG_DEBUG("KV cache receive request %zu, context request %zu started.", requestId, contextRequestId);
        if (llmRequest.getKvCacheTransferStart() == LlmRequest::TimePoint{})
        {
            llmRequest.setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
        }

        std::optional<TransferSession> session;
        try
        {
            if (perRequestCancel.load(std::memory_order_relaxed) || mTerminate.load(std::memory_order_relaxed))
            {
                TLLM_THROW("KV cache receive request %zu cancelled before request-info", requestId);
            }
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            TLLM_LOG_DEBUG("KV cache receive request %zu, context request %zu phase=%s begin.", requestId,
                contextRequestId, phase);
            auto const* cancelFlag = common::getEnvDisaggEnableInflightCancel() ? &perRequestCancel : nullptr;
            session.emplace(sendRequestInfo(llmRequest, cancelFlag));
            session->setTime(TransferSession::kTimeRequestInfo);
            TLLM_LOG_DEBUG(
                "KV cache receive request %zu, context request %zu phase=%s end.", requestId, contextRequestId, phase);

            phase = "ready-signal";
            TLLM_LOG_DEBUG("KV cache receive request %zu, context request %zu phase=%s begin.", requestId,
                contextRequestId, phase);
            auto readyResult = receiveReadySignalDetailed(*session, perRequestCancel);
            TLLM_LOG_DEBUG("KV cache receive request %zu, context request %zu phase=%s end: result=%d.", requestId,
                contextRequestId, phase, static_cast<int>(readyResult));
            if (readyResult == ReadySignalResult::kCancelled)
            {
                if (common::getEnvDisaggEnableInflightCancel())
                {
                    session->poisonReservedRecvBuffers();
                }
                TLLM_THROW("KV cache receive request %zu cancelled while waiting for the ready signal", requestId);
            }
            if (readyResult == ReadySignalResult::kNotReady)
            {
                session->releaseReservedRecvBuffers();
                TLLM_THROW("KV cache receive request %zu was rejected by the context peer", requestId);
            }
            if (readyResult == ReadySignalResult::kMixed)
            {
                if (common::getEnvDisaggEnableInflightCancel())
                {
                    session->poisonReservedRecvBuffers();
                }
                else
                {
                    session->releaseReservedRecvBuffers();
                }
                TLLM_THROW("KV cache receive request %zu received inconsistent ready signals from its context peers",
                    requestId);
            }

            phase = "transfer-completion-notification";
            TLLM_LOG_DEBUG("KV cache receive request %zu, context request %zu phase=%s begin.", requestId,
                contextRequestId, phase);
            receiveSync(*session);
            TLLM_LOG_DEBUG(
                "KV cache receive request %zu, context request %zu phase=%s end.", requestId, contextRequestId, phase);
            llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
        }
        catch (std::exception const& err)
        {
            if (common::getEnvDisaggEnableInflightCancel() && session.has_value())
            {
                session->poisonReservedRecvBuffers();
            }
            llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
            TLLM_LOG_ERROR("KV cache receive request %zu, context request %zu failed in phase=%s: %s", requestId,
                contextRequestId, phase, err.what());
            throw;
        }
        catch (...)
        {
            if (common::getEnvDisaggEnableInflightCancel() && session.has_value())
            {
                session->poisonReservedRecvBuffers();
            }
            llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
            TLLM_LOG_ERROR(
                "KV cache receive request %zu, context request %zu failed in phase=%s with an unknown "
                "exception",
                requestId, contextRequestId, phase);
            throw;
        }

        TLLM_LOG_DEBUG("KV cache receive request %zu, context request %zu completed.", requestId, contextRequestId);
    }

    void requestSync(LlmRequest& llmRequest)
    {
        requestSync(llmRequest, mTerminate);
    }

    struct RequestAndPromise
    {
        // shared_ptr so this struct co-owns the request until the promise resolves;
        // protects worker-side dereferences and the promise itself from premature destruction.
        std::shared_ptr<LlmRequest> mRequest;
        std::unique_ptr<std::promise<void>> mPromise;
        std::shared_ptr<std::atomic<bool>> mCancelFlag;

        RequestAndPromise()
            : mRequest(nullptr)
            , mPromise(nullptr)
            , mCancelFlag(nullptr)
        {
        }

        RequestAndPromise(std::shared_ptr<LlmRequest> request, std::unique_ptr<std::promise<void>>&& promise,
            std::shared_ptr<std::atomic<bool>> cancelFlag)
            : mRequest(std::move(request))
            , mPromise(std::move(promise))
            , mCancelFlag(std::move(cancelFlag))
        {
        }

        RequestAndPromise(RequestAndPromise const&) = delete;

        RequestAndPromise(RequestAndPromise&& other) noexcept
            : mRequest(std::move(other.mRequest))
            , mPromise(std::move(other.mPromise))
            , mCancelFlag(std::move(other.mCancelFlag))
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
                mCancelFlag = std::move(other.mCancelFlag);
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
                    auto const& cancelFlag
                        = requestAndPromise.mCancelFlag != nullptr ? *requestAndPromise.mCancelFlag : mTerminate;
                    requestSync(*requestAndPromise.mRequest, cancelFlag);
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
                catch (...)
                {
                    TLLM_LOG_ERROR("Unknown exception in CacheReceiver request() loop");
                    if (requestAndPromise.mPromise)
                    {
                        requestAndPromise.mPromise->set_exception(std::current_exception());
                    }
                }
                if (common::getEnvDisaggEnableInflightCancel() && requestAndPromise.mRequest != nullptr)
                {
                    std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
                    mInFlightCancelFlags.erase(requestAndPromise.mRequest->mRequestId);
                }
            }
        }
    }

public:
    void setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
        std::vector<SizeType32> rnnLayerNumPerPP, tensorrt_llm::DataType convStateDataType,
        tensorrt_llm::DataType ssmStateDataType)
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
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
    std::mutex mMeasuresFileMutex;
    std::atomic<bool> mTerminate{false};
    std::mutex mInFlightCancelMutex;
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<std::atomic<bool>>> mInFlightCancelFlags;
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
    : CacheSender(manager, selfIndex, std::move(cacheLayer), kMaxPendingPreHandshakeCancellations,
        kPreHandshakeCancellationRetention)
{
}

CacheSender::CacheSender(executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex,
    CacheTransferLayer cacheLayer, std::size_t maxPendingPreHandshakeCancellations,
    std::chrono::milliseconds preHandshakeCancellationRetention)
    : mImpl{std::unique_ptr<Impl, ImplDeleter>(new Impl(manager, selfIndex, std::move(cacheLayer),
        maxPendingPreHandshakeCancellations, preHandshakeCancellationRetention))}
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
    while (true)
    {
        auto requestInfo = mImpl->recvRequestInfo(false);
        TLLM_CHECK(requestInfo.has_value());
        if (!requestInfo->handledWithoutTransfer)
        {
            return std::move(requestInfo->requestInfo);
        }
    }
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
    std::vector<SizeType32> rnnLayerNumPerPP, tensorrt_llm::DataType convStateDataType,
    tensorrt_llm::DataType ssmStateDataType)
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
    std::vector<SizeType32> rnnLayerNumPerPP, tensorrt_llm::DataType convStateDataType,
    tensorrt_llm::DataType ssmStateDataType)
{
    mImpl->setRnnConfig(std::move(rnnModelConfig), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType);
}

} // namespace tensorrt_llm::batch_manager
