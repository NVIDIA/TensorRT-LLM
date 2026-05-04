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
#include <future>
#include <map>
#include <memory>
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
        std::promise<void> promise;
        auto future = promise.get_future();
        llmRequest->setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
        // Register a per-request cancel flag for this sender-side request,
        // symmetric to CacheReceiver::Impl's flag registry. This allows
        // CacheSender::cancelRequest to flip the flag when the request is
        // already being processed (i.e. when it is the mCurrentRequest and
        // therefore not cancellable via the queue-drain path). The flag is
        // consumed by the session's DataContext (built in recvRequestInfo)
        // and by AgentConnection::send's poll-wait loop, which breaks out
        // on cancel.
        (void) getOrCreateInFlightCancelFlag(llmRequest->mRequestId);
        {
            {
                std::scoped_lock lkResp(mSenderMutex);
                // Worker holds shared_ptr so Python-side _terminate_request
                // cannot drop the LlmRequest out from under the async-send
                // worker's dereferences.
                mReadyResponses.emplace(llmRequest->mRequestId, Response{llmRequest, std::move(promise)});
            }
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = true;
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
            // Erase the session first so its DataContext (which references the
            // per-request cancel atomic) is destroyed before we drop the
            // flag's shared_ptr below. This ordering guarantees no dangling
            // reference from DataContext into a freed atomic.
            mRequestToSession.erase(it);
        }
        // Drop the per-request cancel flag now that the session is gone.
        // This is the single point where flags are reclaimed during normal
        // operation; cancel paths that skip release() (sendSync threw or
        // mCancelledRequests fired) intentionally leak the flag shared_ptr
        // until ~Impl / terminate(), matching the existing session-leak
        // behavior on those error paths.
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            mInFlightCancelFlags.erase(requestId);
        }
    }

    [[nodiscard]] RequestInfo recvRequestInfo()
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        bool isAgent = agentConnectionManager != nullptr;

        TransceiverTag::Id id;
        RequestInfo info;
        auto const* connection = isAgent
            ? agentConnectionManager->recvConnectionAndRequestInfo(info, mTerminate)
            : mManager->recvConnect(DataContext{TransceiverTag::kID_TAG, mTerminate}, &id, sizeof(id));
        if (connection == nullptr && !mManager->isRunning())
        {
            TLLM_LOG_WARNING(" recvRequestInfo connection is nullptr, maybe the server is terminating");
            return info;
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
        // Get or create the per-request cancel flag for this requestId. If
        // sendAsync already ran for this reqId the flag is registered; if
        // recvRequestInfo races ahead, we create it here so the session
        // DataContext below holds a live reference. The flag's shared_ptr
        // lifetime is tied to the mInFlightCancelFlags map; it's erased
        // only after sendSync finishes (in removeResponse / cancel
        // cleanup), so the DataContext reference never dangles.
        auto cancelFlag = getOrCreateInFlightCancelFlag(requestId);
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            if (it == mRequestToSession.end())
            {
                // Use the per-request cancel flag (not mTerminate) so a
                // cancel fired by CacheSender::cancelRequest mid-send is
                // observed by AgentConnection::send's poll-wait loop via
                // ctx.getTransferTerminate(). Shutdown still works because
                // ~Impl flips every registered per-request flag before
                // joining the response worker.
                auto session = TransferSession(std::vector<Connection const*>(allCounterparts.size(), nullptr),
                    DataContext{tagFromRequestId(requestId), *cancelFlag}, allCounterparts, mSelfState,
                    info.getTransState(), mBufferManager, info.getIndexFromEnd(), info.getLastBlockKey(), nullptr,
                    !common::getEnvKVCacheTimeOutputPath().empty());
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
            std::scoped_lock lkResp(mSenderMutex);
            auto it = mReadyResponses.find(llmRequest.mRequestId);
            // If the request is not the current request and already in the ready queue, we can cancel it.
            if (it != mReadyResponses.end()
                && (!mCurrentRequest.has_value() || getCurrentRequestId() != llmRequest.mRequestId))
            {
                mCancelledRequests.insert(llmRequest.mRequestId);
                isCancelled = true;
            }
        }
        if (!isCancelled)
        {
            // Request is the mCurrentRequest (or not even in mReadyResponses)
            // — the queue-drain branch can't abort it. Flip the per-request
            // cancel flag registered at sendAsync time; AgentConnection::send
            // observes ctx.getTransferTerminate() in its poll-wait loop and
            // throws, which unwinds sendSync and lets the response worker
            // resume. Symmetric to CacheReceiver::cancelRequest's in-flight
            // cancel-flag branch below.
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            auto flagIt = mInFlightCancelFlags.find(llmRequest.mRequestId);
            if (flagIt != mInFlightCancelFlags.end())
            {
                flagIt->second->store(true);
                isCancelled = true;
                TLLM_LOG_DEBUG("Flipped in-flight sender cancel flag for request %zu (not in queue or is current).",
                    llmRequest.mRequestId);
            }
            else
            {
                TLLM_LOG_WARNING("Cannot cancel request %zu", llmRequest.mRequestId);
            }
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
        // Store shared_ptr rather than raw pointer so the async-send worker's
        // dereferences stay safe past Python-side _terminate_request. Same
        // UAF mitigation as RequestAndPromise on the receiver side.
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
            auto new_exception = TLLM_REQUEST_EXCEPTION(id, e.getErrorCode(), "%s", e.what());
            resp.mPromise.set_exception(std::make_exception_ptr(new_exception));
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s request id: %ld", e.what(), id);
            resp.mPromise.set_exception(std::current_exception());
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
            sendReadySignal(reqId, isReady);

            if (isReady)
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
                    cancelledResponse = std::move(it->second);
                    mReadyResponses.erase(it);
                    mCancelledRequests.erase(cancelledReqId);
                    mRemainSendCount.erase(cancelledReqId);
                }
                // Intentionally do NOT erase mInFlightCancelFlags[cancelledReqId]
                // here — see removeResponse for rationale. The session for
                // this reqId remains in mRequestToSession (release() is not
                // called on this path), so its DataContext still references
                // the atomic. The flag shared_ptr is reclaimed at ~Impl.
                mCurrentRequest = std::nullopt;

                if (mReadyResponses.empty())
                {
                    std::unique_lock lk(mCondMutex);
                    mAnyReady = false;
                }
                cancelledResponse.mPromise.set_exception(std::make_exception_ptr(
                    TLLM_REQUEST_EXCEPTION(cancelledReqId, common::RequestErrorCode::kNETWORK_ERROR,
                        "KV cache transfer for request %zu was cancelled", cancelledReqId)));
            }
        }
        mCurrentRequest = std::nullopt;
    }

    void response() noexcept
    {
        try
        {
            tensorrt_llm::common::setThreadName("dataTransResp");
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            while (!mTerminate || !mAnyReady)
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
                if (!mReadyResponses.empty())
                {
                    auto const& requestInfo = recvRequestInfo();
                    if (mTerminate || !mManager->isRunning())
                    {
                        return;
                    }
                    auto reqId = requestInfo.getRequestId();

                    {
                        std::scoped_lock lk(mSenderMutex);
                        mCurrentRequest = reqId;
                    }

                    if (mRemainSendCount.find(reqId) == mRemainSendCount.end())
                    {
                        mRemainSendCount[reqId] = getCounterpartsCount(reqId);
                    }
                }
                auto it = getCurrentResponse();
                if (it != mReadyResponses.end())
                {
                    sendResponse(it);
                }
                else
                {
                    auto it = getCurrentResponse();
                    while (it == mReadyResponses.end())
                    {
                        std::unique_lock lk(mCondMutex);
                        mSenderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                        if (mTerminate)
                        {
                            break;
                        }
                        it = getCurrentResponse();
                    }
                    sendResponse(it);
                }
            }
        }
        catch (std::exception const& err)
        {
            TLLM_LOG_ERROR("Exception in CacheSender response: %s", err.what());
            for (auto& it : mReadyResponses)
            {
                it.second.mPromise.set_exception(std::current_exception());
            }
        }
        catch (...)
        {
            // Non-std::exception escape (integer throw, custom type throw
            // from NIXL/UCX backends, C++ ABI edge cases). response() is
            // noexcept, so an uncaught non-std throw would call
            // std::terminate. Catch here to resolve pending promises with
            // the exception so callers see a failure instead of the
            // process aborting; the symmetric catch is in
            // CacheReceiver::Impl::request(). The worker thread still
            // exits after this catch — sender is then dead for this
            // process, but fail-closed via the promises is strictly
            // better than terminate().
            TLLM_LOG_ERROR("[CacheSender] UNKNOWN (non-std::exception) escape in response() — worker exiting");
            for (auto& it : mReadyResponses)
            {
                try
                {
                    it.second.mPromise.set_exception(std::current_exception());
                }
                catch (...)
                {
                    // promise already satisfied
                }
            }
        }
    }

    void terminate()
    {
        {
            std::unique_lock lk(mCondMutex);
            mTerminate = true;
        }
        // Flip every registered per-request cancel flag so any AgentConnection::send
        // currently in its poll-wait observes shutdown via the same atomic it
        // polls for per-request cancellation. Needed because recvRequestInfo's
        // TransferSession DataContext references per-request flags (not
        // mTerminate), so without this, shutdown would not interrupt an
        // in-flight sender wedge.
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            for (auto& [id, flag] : mInFlightCancelFlags)
            {
                flag->store(true);
            }
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
    }

    void removeResponse(std::map<RequestIdType, Response>::iterator it)
    {
        {
            std::scoped_lock lkResp(mSenderMutex);
            mReadyResponses.erase(it);
        }
        // Flag erase is intentionally NOT here. The session's DataContext
        // references the atomic held by mInFlightCancelFlags; dropping the
        // map entry while the session still exists in mRequestToSession
        // would dangle that reference. Flag lifetime is tied to session
        // lifetime — reclaimed in release() (called by
        // sendAndRemoveResponse's success path after sendSync completes)
        // or at ~Impl for leaked sessions on error paths.
        if (mReadyResponses.empty())
        {
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = false;
        }
    }

    [[nodiscard]] RequestIdType getCurrentRequestId() const
    {
        return mCurrentRequest.value();
    }

    [[nodiscard]] std::map<RequestIdType, Response>::iterator getCurrentResponse()
    {
        std::scoped_lock lk(mSenderMutex);
        return mReadyResponses.find(getCurrentRequestId());
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
    executor::DataTransceiverState mSelfState;
    CacheTransferLayer mCacheTransferLayer;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
    // Per-request cancel-flag registry (sender-side parity with
    // CacheReceiver::Impl). Registered at sendAsync time, referenced by
    // the TransferSession's DataContext in recvRequestInfo, flipped by
    // cancelRequest on the non-queue-drainable case, and erased after
    // removeResponse / cancel-cleanup. ~terminate flips all for shutdown.
    std::mutex mInFlightCancelMutex;
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<std::atomic<bool>>> mInFlightCancelFlags;
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
        // TODO: Modify the implementation here to avoid frequent thread creation.
        // Lambda captures the shared_ptr so the request is kept alive until the
        // async task completes — closes the raw-pointer UAF that the old
        // `[this, &llmRequest]` capture was vulnerable to.
        auto llmRequestCopy = llmRequest;
        return std::async(std::launch::async, [this, llmRequestCopy]() { requestSync(*llmRequestCopy); });
    }

    [[nodiscard]] std::future<void> requestAndReceiveAsyncMultiThreads(std::shared_ptr<LlmRequest> const& llmRequest)
    {
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
            // Register a per-request cancel flag so cancelRequest() can abort
            // the receive even after it has been dequeued and is blocked inside
            // requestSync (e.g. waiting on recvReadySignal for a peer that
            // will never respond — the ghost-UCX scenario described in the
            // gen-side no-recovery investigation). The shared_ptr keeps the
            // atomic alive for the duration of both the RequestAndPromise
            // (held on the queue / in the worker's local) and any downstream
            // DataContext references.
            auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
            {
                std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
                mInFlightCancelFlags[llmRequest->mRequestId] = cancelFlag;
            }
            {
                std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
                // Worker holds shared_ptr so Python-side _terminate_request
                // cannot drop the LlmRequest out from under the worker's
                // dereferences — closes the raw-pointer UAF.
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

    // Overload kept for the public CacheReceiver::sendRequestInfo wrapper
    // and any direct caller without a per-request cancel flag. Threads
    // mTerminate through so process shutdown still interrupts the poll loop.
    TransferSession sendRequestInfo(LlmRequest const& llmRequest)
    {
        return sendRequestInfo(llmRequest, mTerminate);
    }

    TransferSession sendRequestInfo(LlmRequest const& llmRequest, std::atomic<bool> const& perRequestCancel)
    {
        // Legacy / public-wrapper path: no pre-acquired ids; acquires
        // internally with no RAII protection. Not on the drain-worker path.
        return sendRequestInfo(llmRequest, perRequestCancel, /*preAcquiredCacheBufferIds=*/{});
    }

    // Overload that takes caller-acquired buffer indices (wrapped in
    // BufferIndexHolders at the call site). requestSync uses this variant so
    // RAII covers all non-happy-path exits (not-ready, cancel, throw). The
    // preAcquiredCacheBufferIds vector must have one entry per cache
    // transfer buffer manager (aligned with
    // AgentConnectionManager::getCacheTransBufferManagers()).
    TransferSession sendRequestInfo(LlmRequest const& llmRequest, std::atomic<bool> const& perRequestCancel,
        std::vector<std::optional<size_t>> preAcquiredCacheBufferIds)
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
            auto beam = 0;
            auto const srcPpSize = destCacheState.getParallelConfig().mPipelineParallelism;
            auto requestedBlockRange = getBlockRangeForReceiving(cacheManager, llmRequest,
                destCacheState.getEnableBlockReuse(), destCacheState.getEnablePartialReuse(),
                /*recvSideHasCP=*/false, srcPpSize);

            auto const& uniqueTokens = llmRequest.getUniqueTokens(beam);
            auto lastBlockKey
                = BlockKey(llmRequest.getInputTokensExtraIds().has_value(), llmRequest.getLoraTaskId(), uniqueTokens);
            auto tokensPerBlock = cacheManager->getBlockManager().getTokensPerBlock();
            SizeType32 startTokenIdx = static_cast<SizeType32>(uniqueTokens.size() / tokensPerBlock) * tokensPerBlock;
            SizeType32 endTokenIdx = static_cast<SizeType32>(uniqueTokens.size());
            auto extraKeys = kv_cache_manager::generateBlockHashExtraKeys(llmRequest, startTokenIdx, endTokenIdx);
            lastBlockKey.extraKeys = std::move(extraKeys);
            // Compute indexFromEnd from the number of requested blocks
            int32_t requestedBlockSize = requestedBlockRange.getBlockIdsPerWindow().begin()->second.size();
            TLLM_CHECK_WITH_INFO(requestedBlockSize > 0, "requestedBlockSize must be > 0");
            int32_t indexFromEnd = requestedBlockSize - 1;

            requestInfo = RequestInfo(requestId, mSelfState, indexFromEnd, lastBlockKey);
        }

        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        std::vector<std::optional<size_t>> cacheBufferIds;
        if (agentConnectionManager)
        {
            if (!preAcquiredCacheBufferIds.empty())
            {
                // requestSync already acquired these indices under RAII
                // holders — use them as-is so receiveSync's formatter
                // releases via the existing path while any non-happy-path
                // exit from requestSync releases via ~BufferIndexHolder.
                cacheBufferIds = std::move(preAcquiredCacheBufferIds);
                TLLM_CHECK(!cacheBufferIds.empty());
            }
            else
            {
                // Legacy path: no RAII protection. Acquire internally,
                // matching pre-RAII behavior for callers that don't go
                // through requestSync.
                auto const reqIdForLog = std::make_optional(static_cast<uint64_t>(llmRequest.mRequestId));
                for (auto& cacheTransBufferManager : agentConnectionManager->getCacheTransBufferManagers())
                {
                    cacheBufferIds.push_back(cacheTransBufferManager->assignBufferIndexForRecv(
                        &perRequestCancel, kBufferAcquireSliceMs, reqIdForLog));
                }
                TLLM_CHECK(!cacheBufferIds.empty());
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

        for (size_t ci = 0; ci < allCounterparts.size(); ci++)
        {
            // Honor perRequestCancel between per-peer notify iterations. If
            // the drain worker's cancelRequest has fired (e.g. the gen-side
            // kv_transfer_timeout_ms hit), bail out of this loop instead of
            // continuing to notify peers that have already abandoned their
            // side of the transfer. Without this check the for-body can
            // block on notifySyncMessage to a stuck peer without any
            // opportunity to observe the cancel flag.
            if (perRequestCancel.load(std::memory_order_relaxed))
            {
                TLLM_THROW("sendRequestInfo cancelled via perRequestCancel for request %zu", llmRequest.mRequestId);
            }
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

                const_cast<executor::kv_cache::AgentConnection*>(agentConnection)
                    ->sendRequestAndBufferInfo(requestInfo, idsForRank, validConnectionIdx, &perRequestCancel);
            }
            else
            {
                sendRequestInfo(connection, requestInfo, perRequestCancel);
            }
        }
        auto const& resource = getReceiveCacheResource(llmRequest);
        // The TransferSession's DataContext is used by downstream data-phase
        // operations (receiveSync -> unformat -> per-connection send/recv).
        // Wire perRequestCancel through so those calls observe the same
        // cancel flag the per-iteration loops here honor. AgentConnection::send
        // reads ctx.getTransferTerminate() in its wait-poll loop.
        return TransferSession(std::move(allConnections), DataContext{tagFromRequestId(requestId), perRequestCancel},
            std::move(allCounterparts), mSelfState, contextState, resource->mBufferManager,
            requestInfo.getIndexFromEnd(), requestInfo.getLastBlockKey(), &llmRequest,
            !common::getEnvKVCacheTimeOutputPath().empty());
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

    void sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info,
        std::atomic<bool> const& perRequestCancel)
    {
        std::ostringstream oss;
        RequestInfo::serialize(info, oss);
        auto const& serializedInfo = oss.str();
        std::size_t const infoSize = serializedInfo.size();
        TransceiverTag::Id id{TransceiverTag::Id::REQUEST_SEND};
        // Propagate perRequestCancel via each DataContext so the underlying
        // connection->send implementation (e.g. AgentConnection::send's
        // poll-wait on submitted transfer) can observe a cancel fired
        // asynchronously by CacheReceiver::cancelRequest.
        connection->send(DataContext{TransceiverTag::kID_TAG, perRequestCancel}, &id, sizeof(id));
        connection->send(DataContext{TransceiverTag::kINFO_SIZE_TAG, perRequestCancel}, &infoSize, sizeof(infoSize));
        connection->send(DataContext{TransceiverTag::kINFO_TAG, perRequestCancel}, serializedInfo.data(), infoSize);
    }

    bool cancelRequest(LlmRequest const& llmRequest)
    {

        std::string processInfo = kDefaultProcessInfo;
        if (common::getEnvRequestKVCacheConcurrent())
        {
            processInfo = llmRequest.getDataTransceiverState().getCommState()->toString();
        }

        bool isCancelled = false;
        std::optional<LlmRequest::RequestIdType> queuedCancelledReqId;
        auto& asyncResource = mInstanceToAsyncResource.at(processInfo);
        {
            std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
            auto it = std::find_if(asyncResource->mRequestsQueue.begin(), asyncResource->mRequestsQueue.end(),
                [&llmRequest](RequestAndPromise const& requestAndPromise)
                { return requestAndPromise.mRequest->mRequestId == llmRequest.mRequestId; });
            if (it != asyncResource->mRequestsQueue.end())
            {
                // Fulfil the queued promise with a structured cancellation
                // exception before erasing the entry. Without this, dropping the
                // unique_ptr<promise<void>> destroys it unfulfilled, and any
                // consumer awaiting the corresponding future via
                // mRequesterFutures observes std::future_error: Broken promise
                // instead of a clean per-request kNETWORK_ERROR.
                queuedCancelledReqId = it->mRequest->mRequestId;
                if (it->mPromise)
                {
                    try
                    {
                        it->mPromise->set_exception(std::make_exception_ptr(
                            TLLM_REQUEST_EXCEPTION(*queuedCancelledReqId, common::RequestErrorCode::kNETWORK_ERROR,
                                "Generation KV cache request cancelled before send for request %zu",
                                *queuedCancelledReqId)));
                    }
                    catch (std::future_error const&)
                    {
                        // Promise already satisfied (or no associated future);
                        // nothing else to do.
                    }
                }
                asyncResource->mRequestsQueue.erase(it);
                isCancelled = true;
            }
        }
        if (queuedCancelledReqId.has_value())
        {
            // The worker will never dequeue this entry, so remove the cancel
            // flag here instead of relying on the normal request() cleanup.
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            mInFlightCancelFlags.erase(*queuedCancelledReqId);
        }
        if (!isCancelled)
        {
            // Request already dequeued past mRequestsQueue (worker thread has
            // picked it up; most likely blocked inside requestSync ->
            // recvReadySignal waiting on a peer). Flip the per-request cancel
            // flag so the notification polling loop in
            // AgentConnectionManager::waitForNotification observes it and
            // returns early with isReady=false. requestSync then falls into
            // the kDISAGG_TRANS_ERROR branch and the future resolves, freeing
            // the NIXL/UCX per-request state instead of leaking it.
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            auto flagIt = mInFlightCancelFlags.find(llmRequest.mRequestId);
            if (flagIt != mInFlightCancelFlags.end())
            {
                flagIt->second->store(true);
                isCancelled = true;
                TLLM_LOG_DEBUG("Flipped in-flight cancel flag for request %zu (not in queue).", llmRequest.mRequestId);
            }
            else
            {
                TLLM_LOG_WARNING("Cannot cancel request %zu", llmRequest.mRequestId);
            }
        }
        return isCancelled;
    }

    enum class ReadySignalResult
    {
        kReady,
        kNotReady,
        kCancelled,
    };

    ReadySignalResult receiveReadySignalDetailed(TransferSession& session, std::atomic<bool> const& perRequestCancel)
    {
        bool isReadyFinal = true;
        bool isReady = false;
        auto const& connections = session.getConnections();

        for (size_t i = 0; i < connections.size(); i++)
        {
            // Honor perRequestCancel between per-peer ready-signal waits so
            // a cancel fired during the multi-rank wait sequence bails out
            // rather than continuing to wait on subsequent peers.
            if (perRequestCancel.load(std::memory_order_relaxed))
            {
                return ReadySignalResult::kCancelled;
            }
            auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
            if (agentConnectionManager)
            {
                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(i));
                TLLM_CHECK(agentConnection);
                // Pass the per-request cancel flag as the DataContext's
                // transferTerminate. The notification polling loop in
                // AgentConnectionManager::waitForNotification checks this
                // atomic and returns early when it flips — either on
                // process shutdown (all per-request flags flipped in
                // ~Impl()) or on per-request cancelRequest().
                auto readyResult = agentConnection->recvReadySignalWithStatus(
                    executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG, perRequestCancel});
                if (!readyResult.has_value())
                {
                    return ReadySignalResult::kCancelled;
                }
                isReady = readyResult.value();
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
        }

        return isReadyFinal ? ReadySignalResult::kReady : ReadySignalResult::kNotReady;
    }

    bool receiveReadySignal(TransferSession& session, std::atomic<bool> const& perRequestCancel)
    {
        return receiveReadySignalDetailed(session, perRequestCancel) == ReadySignalResult::kReady;
    }

    // Overload preserved for the (currently unused) std::async-based receiveAsync
    // path and for any callers that don't plumb a per-request cancel flag. Uses
    // the process-level mTerminate only.
    bool receiveReadySignal(TransferSession& session)
    {
        return receiveReadySignal(session, mTerminate);
    }

    ~Impl()
    {
        mTerminate.store(true);
        // Flip every per-request cancel flag so recvReadySignal's polling
        // loop can observe shutdown via the same atomic it uses for
        // per-request cancellation. The shared_ptr aliasing in DataContext
        // keeps these alive for any still-in-flight recvReadySignal calls.
        {
            std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
            for (auto& [id, flag] : mInFlightCancelFlags)
            {
                flag->store(true);
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
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "Start calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
        llmRequest.setKvCacheTransferStart(std::chrono::steady_clock::now());
        // Early-out if cancel fired between queueing and dequeue.
        if (perRequestCancel.load() || mTerminate.load())
        {
            llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());
            return;
        }
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

        // Pre-acquire receive-buffer indices under RAII holders BEFORE entering
        // sendRequestInfo. This is the core of the RAII fix: the formatter
        // inside receiveSync releases the indices on the happy path (hence
        // the `detach()` after a successful receiveSync below), but every
        // OTHER exit path from requestSync used to leak one index per exit
        // (e.g. an early return on `(not-ready or cancel after ready)` would
        // permanently wedge the size-1 pool). With the holder, any return or
        // throw between acquisition and the final detach releases the index
        // via the holder's destructor, closing the class of bug rather than
        // any specific branch.
        std::vector<BufferIndexHolder> recvHolders;
        std::vector<std::optional<size_t>> cacheBufferIds;
        auto* agentConnectionManagerForAcq = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        if (agentConnectionManagerForAcq)
        {
            auto const reqIdForLog = std::make_optional(static_cast<uint64_t>(llmRequest.mRequestId));
            auto const& managers = agentConnectionManagerForAcq->getCacheTransBufferManagers();
            recvHolders.reserve(managers.size());
            cacheBufferIds.reserve(managers.size());
            for (auto* cacheTransBufferManager : managers)
            {
                auto rawIdx = cacheTransBufferManager->assignBufferIndexForRecv(
                    &perRequestCancel, kBufferAcquireSliceMs, reqIdForLog);
                recvHolders.emplace_back(*cacheTransBufferManager, rawIdx, /*isRecv=*/true, reqIdForLog);
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

        auto poisonRecvHolders = [&recvHolders]()
        {
            for (auto& h : recvHolders)
            {
                h.poison();
            }
        };

        try
        {
            auto session = sendRequestInfo(llmRequest, perRequestCancel, std::move(cacheBufferIds));
            session.setTime(TransferSession::kTimeRequestInfo);
            // receiveReadySignal blocks inside AgentConnectionManager::waitForNotification's
            // polling loop until the peer sends the ready notification OR the
            // perRequestCancel flag flips. The result is intentionally
            // tri-state: explicit peer not-ready means no data phase will
            // write into the advertised receive buffers, while local cancel /
            // no-notification means TRT-LLM cannot prove the peer is not still
            // writing and must poison the advertised slots.
            auto readyResult = receiveReadySignalDetailed(session, perRequestCancel);
            if (readyResult == ReadySignalResult::kCancelled)
            {
                poisonRecvHolders();
                llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());
                return;
            }
            if (readyResult == ReadySignalResult::kNotReady)
            {
                llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());
                return;
            }

            receiveSync(session);
            llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());

            // Happy path only: the formatter invoked inside receiveSync already
            // released each buffer index via freeBufferIndexForRecv. Detach the
            // holders so they don't double-release when this stack frame
            // unwinds.
            for (auto& h : recvHolders)
            {
                (void) h.detach();
            }
        }
        catch (...)
        {
            if (agentConnectionManagerForAcq)
            {
                poisonRecvHolders();
            }
            llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());
            throw;
        }

        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "End calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
    }

    // Overload used by the (currently unused) std::async-based
    // Impl::receiveAsync path, where no per-request cancel flag is plumbed.
    // Threads the process-wide mTerminate through the same polling path so
    // shutdown still interrupts it.
    void requestSync(LlmRequest& llmRequest)
    {
        requestSync(llmRequest, mTerminate);
    }

    struct RequestAndPromise
    {
        // Store shared_ptr rather than a raw pointer so the async worker's
        // dereferences stay safe even after Python's _terminate_request drops
        // its own pybind shared_ptr. See CacheTransceiver::mSenderFutures for
        // the full lifetime invariant.
        std::shared_ptr<LlmRequest> mRequest;
        std::unique_ptr<std::promise<void>> mPromise;
        // Per-request cancel flag. Flipped by CacheReceiver::Impl::cancelRequest
        // when a timeout-eviction (in checkGenTransferStatus) or similar
        // wants to abort an in-flight receive that has already been dequeued
        // past the mRequestsQueue stage.
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

        RequestAndPromise(RequestAndPromise&& other) noexcept = default;
        RequestAndPromise& operator=(RequestAndPromise&& other) noexcept = default;
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
            auto const reqId = requestAndPromise.mRequest != nullptr ? requestAndPromise.mRequest->mRequestId
                                                                     : static_cast<LlmRequest::RequestIdType>(0);
            {
                try
                {
                    TLLM_CHECK_WITH_INFO(requestAndPromise.mRequest != nullptr, "requestAndPromise.mRequest is null");
                    TLLM_CHECK_WITH_INFO(
                        requestAndPromise.mCancelFlag != nullptr, "requestAndPromise.mCancelFlag is null");
                    requestSync(*requestAndPromise.mRequest, *requestAndPromise.mCancelFlag);
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
                    // Non-std::exception escapes (e.g. throws of integer /
                    // custom types from NIXL / UCX backends, or the rare C++
                    // ABI abort path) would otherwise kill this worker thread
                    // and leave the mRequestsQueue unserviced forever. When
                    // that happens, the in-flight cancel flag registry keeps
                    // inserting entries, cancelRequest keeps satisfying them
                    // via the queue-removal branch and silently drops them,
                    // and the in-flight cancel-flag branch is never reached —
                    // producing a post-saturation no-recovery state where
                    // every new receive just waits kv_transfer_timeout_ms.
                    // Swallow and continue so the drain loop remains alive;
                    // set the promise so the caller's future resolves with an
                    // error rather than hanging.
                    TLLM_LOG_ERROR(
                        "Non-std::exception escape in CacheReceiver request() loop for reqId=%zu; continuing.", reqId);
                    if (requestAndPromise.mPromise)
                    {
                        try
                        {
                            requestAndPromise.mPromise->set_exception(std::current_exception());
                        }
                        catch (...)
                        {
                            // promise already satisfied; nothing else to do
                        }
                    }
                }
                // Deregister the per-request cancel flag regardless of
                // success / exception. The shared_ptr in requestAndPromise
                // still keeps the atomic alive for any in-flight DataContext
                // references that haven't unwound yet.
                if (requestAndPromise.mRequest != nullptr)
                {
                    std::lock_guard<std::mutex> lg(mInFlightCancelMutex);
                    mInFlightCancelFlags.erase(requestAndPromise.mRequest->mRequestId);
                }
            }
        }
    }

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
    // Per-request cancel flags for in-flight receives that have been dequeued
    // past mRequestsQueue. Registered on enqueue, looked up by cancelRequest
    // for in-flight cancellation, and unregistered after requestSync returns.
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
    return mImpl->recvRequestInfo();
}

bool CacheSender::cancelRequest(LlmRequest const& llmRequest)
{
    return mImpl->cancelRequest(llmRequest);
}

void CacheSender::sendReadySignal(LlmRequest::RequestIdType requestId, bool isReady)
{
    mImpl->sendReadySignal(requestId, isReady);
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

} // namespace tensorrt_llm::batch_manager
