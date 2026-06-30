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
#include "tensorrt_llm/batch_manager/cacheTransferState.h"
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
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

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

namespace
{

class RecvBufferAdvertisementGuard
{
public:
    explicit RecvBufferAdvertisementGuard(std::vector<BufferIndexHolder>& holders) noexcept
        : mHolders{holders}
    {
    }

    RecvBufferAdvertisementGuard(RecvBufferAdvertisementGuard const&) = delete;
    RecvBufferAdvertisementGuard& operator=(RecvBufferAdvertisementGuard const&) = delete;

    ~RecvBufferAdvertisementGuard() noexcept
    {
        if (!mDisarmed && mAdvertisementMayHaveOccurred)
        {
            for (auto& holder : mHolders)
            {
                holder.poison();
            }
        }
    }

    [[nodiscard]] bool* advertisementMayHaveOccurred() noexcept
    {
        return &mAdvertisementMayHaveOccurred;
    }

    void disarm() noexcept
    {
        mDisarmed = true;
    }

private:
    std::vector<BufferIndexHolder>& mHolders;
    bool mAdvertisementMayHaveOccurred{false};
    bool mDisarmed{false};
};

} // namespace

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
        , mInflightCancelEnabled{cacheLayer.isInflightCancelEnabled()}
        , mCacheTransferLayer{std::move(cacheLayer)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
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
        (void) getOrCreateInFlightCancelFlag(llmRequest->mRequestId);
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

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId)
    {
        std::unique_lock<std::mutex> lock(mMtxForMap);
        auto it = mRequestToSession.find(requestId);
        TLLM_CHECK(it != mRequestToSession.end());
        return it->second.getConnections().size();
    }

    void release(LlmRequest::RequestIdType requestId)
    {
        if (!common::getEnvKVCacheTimeOutputPath().empty())
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            if (!mMeasuresFile.is_open())
            {
                auto outputPath = getTransferOutputPath("send");
                mMeasuresFile.open(outputPath);
                TLLM_CHECK_WITH_INFO(
                    mMeasuresFile.is_open(), "Failed to open transfer output file: %s", outputPath.string().c_str());
            }
            it->second.exportMeasure(mMeasuresFile, true);
        }
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            mRequestToSession.erase(it);
        }
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
        catch (...)
        {
            // noexcept cleanup after send failure.
        }
        try
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            mInFlightCancelFlags.erase(requestId);
        }
        catch (...)
        {
            // noexcept cleanup after send failure.
        }
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

        auto requestId = info.getRequestId();
        mCacheTransferLayer.validateSupport(info.getTransState());

        auto allCounterparts = mCacheTransferLayer.computeCounterparts(
            mSelfState.getCommState().value().getSelfIdx(), info.getTransState());

        auto peerSelfIdx = info.getTransState().getCommState()->getSelfIdx();
        int peerIdx = std::distance(
            allCounterparts.begin(), std::find(allCounterparts.begin(), allCounterparts.end(), peerSelfIdx));

        TLLM_CHECK_WITH_INFO(peerIdx < static_cast<int>(allCounterparts.size()),
            "Peer rank %d not found in expected counterparts", peerSelfIdx);
        auto cancelFlag = getOrCreateInFlightCancelFlag(requestId);
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            if (it == mRequestToSession.end())
            {
                auto session = TransferSession(std::vector<Connection const*>(allCounterparts.size(), nullptr),
                    DataContext{tagFromRequestId(requestId), *cancelFlag}, allCounterparts, mSelfState,
                    info.getTransState(), mBufferManager, info.getIndexFromEnd(), info.getLastBlockKey(), nullptr,
                    !common::getEnvKVCacheTimeOutputPath().empty(), mInflightCancelEnabled);
                session.setTime(TransferSession::kTimeRequestInfo);
                it = mRequestToSession.emplace(requestId, std::move(session)).first;
            }
            it->second.setConnection(peerIdx, connection);
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
        {
            std::scoped_lock lock(mSenderMutex);
            auto it = mReadyResponses.find(llmRequest.mRequestId);
            // Until ready=true is committed to the peer, cancellation can still
            // reject the request cleanly without starting a transfer.
            if (it != mReadyResponses.end()
                && (!mReadyCommittedRequest.has_value() || mReadyCommittedRequest.value() != llmRequest.mRequestId))
            {
                mCancelledRequests.insert(llmRequest.mRequestId);
                isCancelled = true;
            }
        }
        if (!isCancelled && mInflightCancelEnabled)
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
                agentConnection->sendReadySignal(
                    executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG}, isReady);
            }
            else
            {
                connections.at(i)->send(
                    executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG}, &isReady, sizeof(isReady));
            }
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
            resp.mPromise.set_exception(std::make_exception_ptr(new_exception));
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s request id: %ld", e.what(), id);
            discardTransferState(id);
            resp.mPromise.set_exception(std::current_exception());
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
            failResponse(resp, std::current_exception());
        }
    }

    void sendResponse(RequestIdType reqId)
    {
        bool isReady = true;
        {
            std::scoped_lock lock(mSenderMutex);
            TLLM_CHECK(mCurrentRequest.has_value() && mCurrentRequest.value() == reqId);
            TLLM_CHECK(mReadyResponses.find(reqId) != mReadyResponses.end());
            auto countIt = mRemainSendCount.find(reqId);
            TLLM_CHECK(countIt != mRemainSendCount.end());
            auto const count = --countIt->second;
            TLLM_CHECK(count >= 0);
            if (count > 0)
            {
                mCurrentRequest = std::nullopt;
                return;
            }
            mRemainSendCount.erase(countIt);
            isReady = mCancelledRequests.find(reqId) == mCancelledRequests.end();
            if (isReady)
            {
                mReadyCommittedRequest = reqId;
            }
        }

        // Keep mCurrentRequest set while notifying the peer so cancellation cannot change the decision after it has
        // been made. The network operation must not run under mSenderMutex.
        sendReadySignal(reqId, isReady);

        Response response;
        {
            std::scoped_lock lock(mSenderMutex);
            auto it = mReadyResponses.find(reqId);
            TLLM_CHECK(it != mReadyResponses.end());
            response = std::move(it->second);
            mReadyResponses.erase(it);
            mCancelledRequests.erase(reqId);
            mReadyCommittedRequest = std::nullopt;
            mCurrentRequest = std::nullopt;
        }

        if (isReady)
        {
            if (dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager) != nullptr)
            {
                // Our NIXL implementation only supports recv and send in the same thread. Using ZMQ for the control
                // path may avoid this limitation.
                sendAndRemoveResponse(reqId, std::move(response));
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
            response.mPromise.set_exception(std::make_exception_ptr(TLLM_REQUEST_EXCEPTION(reqId,
                common::RequestErrorCode::kNETWORK_ERROR, "KV cache transfer for request %zu was cancelled", reqId)));
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
            while (true)
            {
                {
                    std::unique_lock lock(mSenderMutex);
                    mSenderCv.wait(lock, [this]() { return mTerminate || !mReadyResponses.empty(); });
                    if (mTerminate)
                    {
                        break;
                    }
                }

                auto requestInfo = recvRequestInfo();
                if (!requestInfo.has_value() || mTerminate || !mManager->isRunning())
                {
                    break;
                }
                auto const reqId = requestInfo->getRequestId();

                if (mRemainSendCount.find(reqId) == mRemainSendCount.end())
                {
                    mRemainSendCount[reqId] = getCounterpartsCount(reqId);
                }

                {
                    std::unique_lock lock(mSenderMutex);
                    mCurrentRequest = reqId;
                    mSenderCv.wait(lock,
                        [this, reqId]() { return mTerminate || mReadyResponses.find(reqId) != mReadyResponses.end(); });
                    if (mTerminate)
                    {
                        mCurrentRequest = std::nullopt;
                        mReadyCommittedRequest = std::nullopt;
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
            mReadyCommittedRequest = std::nullopt;
            mCancelledRequests.clear();
            mRemainSendCount.clear();
        }
        for (auto& entry : pendingResponses)
        {
            failResponse(entry.second, exception);
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
    std::optional<RequestIdType> mCurrentRequest;
    std::optional<RequestIdType> mReadyCommittedRequest;
    std::set<LlmRequest::RequestIdType> mCancelledRequests;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mSenderMutex;
    std::atomic<bool> mTerminate{false};
    std::condition_variable mSenderCv;
    std::future<void> mResponseFuture;
    std::unordered_map<LlmRequest::RequestIdType, int> mRemainSendCount;
    AsyncSendResource mAsyncSendResource;
    std::vector<std::future<void>> mAsyncSendFutures;
    int mDeviceId{-1};

    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, TransferSession> mRequestToSession;
    executor::DataTransceiverState mSelfState;
    bool mInflightCancelEnabled{false};
    CacheTransferLayer mCacheTransferLayer;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
    std::mutex mInFlightCancelMutex;
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<std::atomic<bool>>> mInFlightCancelFlags;
};

class CacheReceiver::Impl
{
public:
    Impl(executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
        : mManager{manager}
        , mSelfState{cacheLayer.getCacheState(), executor::kv_cache::CommState{manager->getCommState()}}
        , mInflightCancelEnabled{cacheLayer.isInflightCancelEnabled()}
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
        return std::async(std::launch::async,
            [this, llmRequestCopy]()
            {
                if (!requestSync(*llmRequestCopy) && mInflightCancelEnabled)
                {
                    throw TLLM_REQUEST_EXCEPTION(llmRequestCopy->mRequestId, common::RequestErrorCode::kNETWORK_ERROR,
                        "Generation KV cache transfer failed for request %zu", llmRequestCopy->mRequestId);
                }
            });
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
            auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
            {
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
            session.releaseRecvBufferHolders();
        }
        catch (...)
        {
            session.poisonRecvBufferHolders();
            throw;
        }
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

    TransferSession sendRequestInfo(LlmRequest const& llmRequest, std::atomic<bool> const* perRequestCancel = nullptr,
        std::vector<std::optional<size_t>> cacheBufferIds = {},
        RecvBufferAdvertisementGuard* externalRecvBufferGuard = nullptr)
    {
        TLLM_CHECK_WITH_INFO(externalRecvBufferGuard == nullptr || !cacheBufferIds.empty(),
            "An external receive-buffer advertisement guard requires preassigned buffer IDs.");
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
        std::vector<BufferIndexHolder> sessionRecvHolders;
        if (agentConnectionManager)
        {
            if (cacheBufferIds.empty())
            {
                auto const& managers = agentConnectionManager->getCacheTransBufferManagers();
                sessionRecvHolders.reserve(managers.size());
                for (auto* cacheTransBufferManager : managers)
                {
                    auto rawIdx = cacheTransBufferManager->assignBufferIndexForRecv(perRequestCancel);
                    sessionRecvHolders.emplace_back(*cacheTransBufferManager, rawIdx, /*isRecv=*/true);
                    if (rawIdx.has_value())
                    {
                        cacheBufferIds.push_back(static_cast<size_t>(rawIdx.value()));
                    }
                    else
                    {
                        cacheBufferIds.push_back(std::nullopt);
                    }
                }
            }
            TLLM_CHECK(!cacheBufferIds.empty());
        }

        RecvBufferAdvertisementGuard sessionRecvBufferGuard{sessionRecvHolders};
        auto& recvBufferGuard = externalRecvBufferGuard != nullptr ? *externalRecvBufferGuard : sessionRecvBufferGuard;
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

        bool sentAgentRequestInfo = false;
        for (size_t ci = 0; ci < allCounterparts.size(); ci++)
        {
            auto rank = allCounterparts[ci];
            auto const* connection = connections.at(rank);

            bool isKvCounterpart
                = std::find(kvCounterParts.begin(), kvCounterParts.end(), rank) != kvCounterParts.end();
            bool isRnnCounterpart
                = hasRnn && std::find(rnnCounterParts.begin(), rnnCounterParts.end(), rank) != rnnCounterParts.end();

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
                    auto [pickUpIdx, localRankIdx] = cache_formatter_utils::pickRecvConnections(rnnCounterParts.size(),
                        mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx(),
                        destCacheState, rnnCounterParts, rnnTargetInfo);
                    validConnectionIdx
                        = std::find(localRankIdx.begin(), localRankIdx.end(), rnnCpIdx) - localRankIdx.begin();
                }

                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection);
                TLLM_CHECK(agentConnection != nullptr);

                // Cancellation may abort before the first notification. Once one
                // counterpart has received the request, finish the fanout so every
                // sender can reach its expected terminal response count.
                auto const* fanoutCancel = sentAgentRequestInfo ? nullptr : perRequestCancel;
                const_cast<executor::kv_cache::AgentConnection*>(agentConnection)
                    ->sendRequestAndBufferInfo(requestInfo, idsForRank, validConnectionIdx, fanoutCancel,
                        recvBufferGuard.advertisementMayHaveOccurred());
                sentAgentRequestInfo = true;
            }
            else
            {
                sendRequestInfo(connection, requestInfo);
            }
        }
        auto const& resource = getReceiveCacheResource(llmRequest);
        TransferSession session = perRequestCancel != nullptr
            ? TransferSession(std::move(allConnections), DataContext{tagFromRequestId(requestId), *perRequestCancel},
                std::move(allCounterparts), mSelfState, contextState, resource->mBufferManager,
                requestInfo.getIndexFromEnd(), requestInfo.getLastBlockKey(), &llmRequest,
                !common::getEnvKVCacheTimeOutputPath().empty(), mInflightCancelEnabled)
            : TransferSession(std::move(allConnections), DataContext{tagFromRequestId(requestId), mTerminate},
                std::move(allCounterparts), mSelfState, contextState, resource->mBufferManager,
                requestInfo.getIndexFromEnd(), requestInfo.getLastBlockKey(), &llmRequest,
                !common::getEnvKVCacheTimeOutputPath().empty(), mInflightCancelEnabled);
        session.adoptRecvBufferHolders(std::move(sessionRecvHolders));
        if (externalRecvBufferGuard == nullptr)
        {
            sessionRecvBufferGuard.disarm();
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
        if (queuedCancelledReqId.has_value())
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            mInFlightCancelFlags.erase(*queuedCancelledReqId);
        }
        if (!isCancelled && mInflightCancelEnabled)
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
        kPartiallyReady,
        kCancelled,
    };

    ReadySignalResult receiveReadySignalDetailed(TransferSession& session, std::atomic<bool> const& perRequestCancel)
    {
        bool isReadyFinal = true;
        bool anyReady = false;
        bool isReady = false;
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
                    executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG, perRequestCancel});
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
            isReadyFinal &= isReady;
            anyReady |= isReady;
        }

        switch (detail::summarizeReadySignals(isReadyFinal, anyReady))
        {
        case detail::ReadySignalSummary::kReady: return ReadySignalResult::kReady;
        case detail::ReadySignalSummary::kNotReady: return ReadySignalResult::kNotReady;
        case detail::ReadySignalSummary::kPartiallyReady: return ReadySignalResult::kPartiallyReady;
        }
        TLLM_THROW("Unknown ready-signal summary");
    }

    bool receiveReadySignal(TransferSession& session)
    {
        auto const result = receiveReadySignalDetailed(session, mTerminate);
        if (result == ReadySignalResult::kPartiallyReady || result == ReadySignalResult::kCancelled)
        {
            session.poisonRecvBufferHolders();
        }
        else if (result == ReadySignalResult::kNotReady)
        {
            // No peer accepted the request, so no writer can have started.
            session.releaseRecvBufferHolders();
        }
        return result == ReadySignalResult::kReady;
    }

    ~Impl()
    {
        mTerminate.store(true);
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
    [[nodiscard]] bool requestSync(LlmRequest& llmRequest, std::atomic<bool> const& perRequestCancel)
    {
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "Start calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
        if (!mInflightCancelEnabled)
        {
            llmRequest.setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
        }
        if (perRequestCancel.load(std::memory_order_relaxed) || mTerminate.load(std::memory_order_relaxed))
        {
            if (!mInflightCancelEnabled)
            {
                llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            }
            llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
            return false;
        }
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

        std::vector<BufferIndexHolder> recvHolders;
        std::vector<std::optional<size_t>> cacheBufferIds;
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        if (agentConnectionManager)
        {
            auto const& managers = agentConnectionManager->getCacheTransBufferManagers();
            recvHolders.reserve(managers.size());
            cacheBufferIds.reserve(managers.size());
            for (auto* cacheTransBufferManager : managers)
            {
                auto rawIdx = cacheTransBufferManager->assignBufferIndexForRecv(&perRequestCancel);
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
        }

        RecvBufferAdvertisementGuard recvBufferGuard{recvHolders};

        try
        {
            auto* externalRecvBufferGuard = agentConnectionManager != nullptr ? &recvBufferGuard : nullptr;
            auto session
                = sendRequestInfo(llmRequest, &perRequestCancel, std::move(cacheBufferIds), externalRecvBufferGuard);
            session.setTime(TransferSession::kTimeRequestInfo);
            auto readyResult = receiveReadySignalDetailed(session, perRequestCancel);
            if (readyResult == ReadySignalResult::kCancelled)
            {
                if (!mInflightCancelEnabled)
                {
                    llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                }
                llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
                return false;
            }
            if (readyResult == ReadySignalResult::kNotReady)
            {
                // Every peer declined the request, so no advertised buffer can have a writer.
                recvBufferGuard.disarm();
                if (!mInflightCancelEnabled)
                {
                    llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                }
                llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
                return false;
            }
            if (readyResult == ReadySignalResult::kPartiallyReady)
            {
                // A peer that answered ready may start writing immediately. Since another peer
                // declined the request, no receive path will prove all advertised buffers
                // quiescent; quarantine them rather than making them available for reuse.
                if (!mInflightCancelEnabled)
                {
                    llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                }
                llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
                return false;
            }

            receiveSync(session);
            llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());

            for (auto& holder : recvHolders)
            {
                holder.release();
            }
            recvBufferGuard.disarm();
        }
        catch (...)
        {
            if (!mInflightCancelEnabled)
            {
                llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            }
            llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
            throw;
        }

        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "End calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
        return true;
    }

    [[nodiscard]] bool requestSync(LlmRequest& llmRequest)
    {
        return requestSync(llmRequest, mTerminate);
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
                    TLLM_CHECK_WITH_INFO(
                        requestAndPromise.mCancelFlag != nullptr, "requestAndPromise.mCancelFlag is null");
                    if (requestSync(*requestAndPromise.mRequest, *requestAndPromise.mCancelFlag)
                        || !mInflightCancelEnabled)
                    {
                        requestAndPromise.mPromise->set_value();
                    }
                    else
                    {
                        requestAndPromise.mPromise->set_exception(std::make_exception_ptr(TLLM_REQUEST_EXCEPTION(
                            requestAndPromise.mRequest->mRequestId, common::RequestErrorCode::kNETWORK_ERROR,
                            "Generation KV cache transfer failed for request %zu",
                            requestAndPromise.mRequest->mRequestId)));
                    }
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
                if (requestAndPromise.mRequest != nullptr)
                {
                    std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
                    mInFlightCancelFlags.erase(requestAndPromise.mRequest->mRequestId);
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
    bool mInflightCancelEnabled{false};
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
