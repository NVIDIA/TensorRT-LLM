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

#include "cacheFormatter.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <future>
#include <map>
#include <memory>
#include <unordered_map>

namespace tensorrt_llm::batch_manager
{

using runtime::SizeType32;

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState)
    : mRequestId{requestId}
    , mTransState{std::move(transState)}
{
}

RequestInfo::RequestInfo(
    LlmRequest::RequestIdType requestId, std::vector<size_t> blockHashes, executor::DataTransceiverState transState)
    : mRequestId{requestId}
    , mBlockHashes{std::move(blockHashes)}
    , mTransState{std::move(transState)}
{
}

bool RequestInfo::operator==(RequestInfo const& rhs) const
{
    return mRequestId == rhs.mRequestId && mBlockHashes == rhs.mBlockHashes && mTransState == rhs.mTransState;
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
    su::serialize(requestInfo.mBlockHashes, os);
    su::serialize(requestInfo.mTransState, os);
}

RequestInfo RequestInfo::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto requestId = su::deserialize<decltype(mRequestId)>(is);
    auto blockHashes = su::deserialize<decltype(mBlockHashes)>(is);
    auto transState = su::deserialize<decltype(mTransState)>(is);
    return RequestInfo{requestId, std::move(blockHashes), std::move(transState)};
}

std::size_t RequestInfo::serializedSize(RequestInfo const& requestInfo)
{
    namespace su = executor::serialize_utils;
    std::size_t totalSize = 0;
    totalSize += su::serializedSize(requestInfo.mRequestId);
    totalSize += su::serializedSize(requestInfo.mBlockHashes);
    totalSize += su::serializedSize(requestInfo.mTransState);
    return totalSize;
}

static int32_t tagFromRequestId(LlmRequest::RequestIdType requestId)
{
    constexpr int32_t kDATA_TAG{43};
    return ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
}

using BaseCacheFormatter = kv_cache_manager::BaseCacheFormatter;

DataSender::DataSender(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
    SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mManager{manager}
    , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
    , mFormatter(std::move(formatter))
    , mBufferManager{std::make_shared<runtime::CudaStream>()}
{
    TLLM_CHECK(mManager);
    TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
}

[[nodiscard]] RequestInfo DataSender::recvRequestInfo()
{
    using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
    auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
    bool isAgent = agentConnectionManager != nullptr;

    auto agentRecvFun = [&](RequestInfo& requestInfo)
    {
        auto const* connection = agentConnectionManager->recvConnectionAndRequestInfo(requestInfo);
        return connection;
    };
    Id id;
    RequestInfo info;
    auto const* connection
        = isAgent ? agentRecvFun(info) : mManager->recvConnect(DataContext{kID_TAG}, &id, sizeof(id));
    if (!isAgent)
    {
        TLLM_CHECK(id == Id::REQUEST_SEND);
        std::uint64_t infoSize{0};
        connection->recv(executor::kv_cache::DataContext{kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
        std::string serializedInfo;
        serializedInfo.resize(infoSize);
        connection->recv(executor::kv_cache::DataContext{kINFO_TAG}, serializedInfo.data(), infoSize);
        std::istringstream iss(serializedInfo);
        info = RequestInfo::deserialize(iss);
    }

    auto requestId = info.getRequestId();
    TLLM_CHECK_WITH_INFO(
        mFormatter->inquireSupport(mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
        "Disagg server does not currently support these cacheState, please check the cacheState of the context and gen "
        "executors");
    auto peerRelativeRanks = executor::kv_cache::targetIRanks(info.getTransState().getCacheState().value(),
        mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx())
                                 .mIRanks;
    int peerIdx = std::distance(peerRelativeRanks.begin(),
        std::find(
            peerRelativeRanks.begin(), peerRelativeRanks.end(), info.getTransState().getCommState()->getSelfIdx()));
    {
        std::unique_lock<std::mutex> lk(mMtxForMap);
        auto it = mRequestToComms.find(requestId);
        if (it == mRequestToComms.end())
        {
            int recvExpectCount = peerRelativeRanks.size();
            {
                it = mRequestToComms.emplace(requestId, RequestMapInfo{}).first;
                it->second.resize(recvExpectCount);
            }
        }
        it->second[peerIdx] = {connection, info.getTransState()};
    }
    return info;
}

void DataSender::sendSync(LlmRequest const& llmRequest)
{
    std::vector<executor::kv_cache::Connection const*> connections;
    auto it = mRequestToComms.find(llmRequest.mRequestId);
    TLLM_CHECK(it != mRequestToComms.end());
    auto const& reqToComm = it->second;
    for (auto&& [connection, dataTransceiverState] : reqToComm)
    {
        connections.emplace_back(connection);
    }
    auto&& dataTransceiverState = reqToComm.at(0).second;
    TransferSession session(connections, DataContext{tagFromRequestId(llmRequest.mRequestId)}, mSelfState,
        dataTransceiverState, mBufferManager);
    mFormatter->format(session, llmRequest);
}

[[nodiscard]] executor::kv_cache::CommState const& DataSender::getCommState() const
{
    return mSelfState.getCommState().value();
}

[[nodiscard]] size_t DataSender::getCounterpartsCount(LlmRequest::RequestIdType requestId) const
{
    auto it = mRequestToComms.find(requestId);
    TLLM_CHECK(it != mRequestToComms.end());
    return it->second.size();
}

void DataSender::release(LlmRequest::RequestIdType requestId)
{
    auto it = mRequestToComms.find(requestId);
    TLLM_CHECK(it != mRequestToComms.end());
    std::unique_lock<std::mutex> lk(mMtxForMap);
    mRequestToComms.erase(it);
}

DataSender::~DataSender() = default;

DataReceiver::DataReceiver(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mManager{manager}
    , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
    , mFormatter(std::move(formatter))
{
    TLLM_CHECK(mManager);
    TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
    TLLM_CHECK(mFormatter);
}

void DataReceiver::sendRequestInfo(LlmRequest const& llmRequest)
{
    uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& commState = contextState.getCommState().value();
    auto const& destCacheState = contextState.getCacheState().value();
    TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
        "Disagg server does not currently support these cacheState.");

    RequestInfo requestInfo(requestId, mSelfState);

    if (!common::getEnvDisableSelectiveCacheTransfer())
    {
        auto* cacheManager = mFormatter->getCacheManager();
        auto blockRange
            = kv_cache_manager::BlockRange::fromNewlyAllocatedBlockIds(*cacheManager, llmRequest.mRequestId);
        requestInfo = RequestInfo(requestId, blockRange.getBlockHashes(), mSelfState);
    }

    auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
    std::optional<size_t> cacheBufferId = std::nullopt;
    if (agentConnectionManager != nullptr)
    {
        cacheBufferId = agentConnectionManager->getCacheTransBufferManager()->assignBufferIndexForRecv();
        TLLM_CHECK(cacheBufferId.has_value());
        // memory Desp , validSegmentIdx send
    }
    auto counterParts = mFormatter->getCounterparts(
        mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState);

    auto connections = mManager->getConnections(commState);
    std::vector<executor::kv_cache::Connection const*> counterPartConnections;
    for (auto index : counterParts)
    {
        auto const* connection = connections.at(index);
        counterPartConnections.emplace_back(connection);
    }
    auto pickUpConnections = mFormatter->pickRecvConnections(counterPartConnections, mSelfState.getCacheState().value(),
        mSelfState.getCommState().value().getSelfIdx(), destCacheState);
    for (auto connection : counterPartConnections)
    {
        // if Manager is agentConnectionManager, then send request info to agent
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        if (agentConnectionManager != nullptr)
        {
            // TODO: index -> validConnectionIdx conversion
            auto valideConnectionIdx
                = std::find(pickUpConnections.begin(), pickUpConnections.end(), connection) - pickUpConnections.begin();
            auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection);
            TLLM_CHECK(agentConnection != nullptr);
            TLLM_CHECK(cacheBufferId.has_value());
            const_cast<executor::kv_cache::AgentConnection*>(agentConnection)
                ->sendRequestAndBufferInfo(requestInfo, cacheBufferId, valideConnectionIdx);
        }
        else
        {
            sendRequestInfo(connection, requestInfo);
        }
    }
}

void DataReceiver::receiveSync(LlmRequest const& llmRequest)
{
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& commState = contextState.getCommState().value();
    auto const& destCacheState = contextState.getCacheState().value();
    std::vector<tensorrt_llm::executor::kv_cache::Connection const*> connections;
    for (auto index : mFormatter->getCounterparts(
             mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
    {
        auto const* connection = mManager->getConnections(commState).at(index);
        connections.emplace_back(connection);
    }
    auto const& resource = getReceiveCacheResource(llmRequest);
    TransferSession session(connections, DataContext{tagFromRequestId(llmRequest.mRequestId)}, mSelfState, contextState,
        resource->mBufferManager);
    mFormatter->unformat(session, llmRequest);
}

void DataReceiver::sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info)
{
    std::ostringstream oss;
    RequestInfo::serialize(info, oss);
    auto const& serializedInfo = oss.str();
    std::size_t const infoSize = serializedInfo.size();
    Id id{Id::REQUEST_SEND};
    connection->send(executor::kv_cache::DataContext{kID_TAG}, &id, sizeof(id));
    connection->send(executor::kv_cache::DataContext{kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
    connection->send(executor::kv_cache::DataContext{kINFO_TAG}, serializedInfo.data(), infoSize);
}

std::unique_ptr<DataReceiver::ReceiveCacheResource> const& DataReceiver::getReceiveCacheResource(
    LlmRequest const& llmRequest)
{
    std::scoped_lock<std::mutex> lock(mProcessIoResouceMutex);
    TLLM_CHECK(llmRequest.getDataTransceiverState().getCommState().has_value());
    std::string processString = "default";
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

class DataResponder::Impl
{
public:
    using RequestIdType = LlmRequest::RequestIdType;

    Impl(std::unique_ptr<DataSender> sender)
        : mSender{std::move(sender)}
    {
        TLLM_CHECK(mSender);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
        mCurrentRequest = std::nullopt;
        mResponseFuture = std::async(std::launch::async, &Impl::response, this);
    }

    [[nodiscard]] std::future<void> respondAndSendAsync(LlmRequest& llmRequest)
    {
        std::promise<void> promise;
        auto future = promise.get_future();
        {
            {
                std::unique_lock lkResp(mResponderMutex);
                mReadyResponses.emplace(
                    llmRequest.mRequestId, Response{std::addressof(llmRequest), std::move(promise)});
            }
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = true;
        }
        mResponderCv.notify_all();
        return future;
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const
    {
        return mSender->getCommState();
    }

    ~Impl()
    {
        terminate();
    }

private:
    struct Response
    {
        LlmRequest* mRequest;
        std::promise<void> mPromise;
    };

    void sendAndRemoveResponse(RequestIdType id, Response resp) noexcept
    {
        try
        {
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            mSender->sendSync(*resp.mRequest);
            mSender->release(id);
            resp.mPromise.set_value();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s ", e.what());
            resp.mPromise.set_exception(std::current_exception());
        }
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
                    mResponderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                }
                if (mTerminate)
                {
                    break;
                }
                std::vector<size_t> blockHashes;
                if (!isSending() && !mReadyResponses.empty())
                {
                    auto const& requestInfo = mSender->recvRequestInfo();
                    auto reqId = requestInfo.getRequestId();
                    blockHashes = requestInfo.getBlockHashes();

                    mCurrentRequest = reqId;
                    if (mRemainSendCount.find(reqId) == mRemainSendCount.end())
                    {
                        mRemainSendCount[reqId] = mSender->getCounterpartsCount(reqId);
                    }
                }
                auto it = getCurrentResponse();
                if (it != mReadyResponses.end())
                {
                    auto reqId = mCurrentRequest.value();
                    auto count = --mRemainSendCount[reqId];
                    TLLM_CHECK(count >= 0);
                    if (count == 0)
                    {
                        mRemainSendCount.erase(reqId);

                        // TODO(zhengd): pass the hashes directly instead of update llmRequest
                        auto llmRequest = it->second.mRequest;
                        llmRequest->setRequestedBlockHashes(std::move(blockHashes));

                        if (common::getEnvParallelCacheSend())
                        {
                            // TODO: Use a thread pool and check for thread safety.
                            std::thread(
                                &DataResponder::Impl::sendAndRemoveResponse, this, it->first, std::move(it->second))
                                .detach();
                        }
                        else
                        {
                            DataResponder::Impl::sendAndRemoveResponse(it->first, std::move(it->second));
                        }
                        removeResponse(it);
                    }
                    mCurrentRequest = std::nullopt;
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(!mCurrentRequest.has_value(),
                        "This executor does not have a prepared KV cache for request ID: %zu, and the "
                        "mReadyResponses size is: %zu. mpi rank :%d     ",
                        mCurrentRequest.value(), mReadyResponses.size(), mpi::MpiComm::world().getRank());
                    std::unique_lock lk(mCondMutex);
                    mResponderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                }
            }
        }
        catch (std::exception const& err)
        {
            TLLM_LOG_ERROR("Exception in DataResponder response: %s", err.what());
            for (auto& it : mReadyResponses)
            {
                it.second.mPromise.set_exception(std::current_exception());
            }
        }
    }

    void terminate()
    {
        {
            std::unique_lock lk(mCondMutex);
            mTerminate = true;
        }
        // We don't have to wait for the future. If another thread is sending data, it won't pay attention
        // to the terminate flag.
        mResponderCv.notify_all();
    }

    void removeResponse(std::map<RequestIdType, Response>::iterator it)
    {
        {
            std::unique_lock lkResp(mResponderMutex);
            mReadyResponses.erase(it);
        }
        if (mReadyResponses.empty())
        {
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = false;
        }
    }

    [[nodiscard]] bool isSending() const
    {
        return mCurrentRequest.has_value();
    }

    [[nodiscard]] RequestIdType getCurrentRequestId() const
    {
        return mCurrentRequest.value();
    }

    [[nodiscard]] std::map<RequestIdType, Response>::iterator getCurrentResponse()
    {
        std::unique_lock lk(mResponderMutex);
        return mReadyResponses.find(getCurrentRequestId());
    }

private:
    std::optional<RequestIdType> mCurrentRequest;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mResponderMutex, mCondMutex;
    std::atomic<bool> mAnyReady{false}, mTerminate{false};
    std::condition_variable mResponderCv;
    std::future<void> mResponseFuture;
    std::unique_ptr<DataSender> mSender;
    std::unordered_map<LlmRequest::RequestIdType, int> mRemainSendCount;
    int mDeviceId{-1};
};

class DataRequester::Impl
{
public:
    Impl(std::unique_ptr<DataReceiver> receiver)
        : mReceiver{std::move(receiver)}
    {
        TLLM_CHECK(mReceiver);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    }

    [[nodiscard]] std::future<void> requestAndReceiveAsync(LlmRequest& llmRequest)
    {
        // TODO: Modify the implementation here to avoid frequent thread creation.
        return std::async(std::launch::async, &DataRequester::Impl::requestSync, this, std::ref(llmRequest));
    }

    [[nodiscard]] std::future<void> requestAndReceiveAsyncMultiThreads(LlmRequest& llmRequest)
    {
        try
        {
            auto promise = std::make_unique<std::promise<void>>();
            auto future = promise->get_future();
            TLLM_CHECK(llmRequest.getDataTransceiverState().getCommState().has_value());
            std::string processInfo = "default";
            if (common::getEnvRequestKVCacheConcurrent())
            {
                processInfo = llmRequest.getDataTransceiverState().getCommState()->toString();
            }
            if (mInstanceToAsyncResource.find(processInfo) == mInstanceToAsyncResource.end())
            {

                mInstanceToAsyncResource.emplace(processInfo, std::make_unique<AsyncResource>());
                auto requestFuture = std::async(std::launch::async, &DataRequester::Impl::request, this,
                    std::ref(*mInstanceToAsyncResource.at(processInfo)));
                mRequestFutures.emplace_back(std::move(requestFuture));
            }
            auto& asyncResource = mInstanceToAsyncResource.at(processInfo);
            {
                std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
                asyncResource->mRequestsQueue.emplace_back(std::addressof(llmRequest), std::move(promise));
            }
            asyncResource->mCVforQueue.notify_all();
            return future;
        }
        catch (std::exception const& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    ~Impl()
    {
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
        llmRequest.setKvCacheTransferStart(std::chrono::steady_clock::now());
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
        mReceiver->sendRequestInfo(llmRequest);
        mReceiver->receiveSync(llmRequest);
        llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());

        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "End calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
    }

    struct RequestAndPromise
    {
        LlmRequest* mRequest;
        std::unique_ptr<std::promise<void>> mPromise;

        RequestAndPromise()
            : mRequest(nullptr)
            , mPromise(nullptr)
        {
        }

        RequestAndPromise(LlmRequest* request, std::unique_ptr<std::promise<void>>&& promise)
            : mRequest(request)
            , mPromise(std::move(promise))
        {
        }

        RequestAndPromise(RequestAndPromise const&) = delete;

        RequestAndPromise(RequestAndPromise&& other) noexcept
            : mRequest(other.mRequest)
            , mPromise(std::move(other.mPromise))
        {
            other.mRequest = nullptr;
        }

        RequestAndPromise& operator=(RequestAndPromise&& other) noexcept
        {
            if (this != &other)
            {
                mRequest = nullptr;
                if (mPromise)
                {
                    mPromise.reset();
                }

                mRequest = other.mRequest;
                mPromise = std::move(other.mPromise);

                other.mRequest = nullptr;
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
                catch (std::exception const& err)
                {
                    TLLM_LOG_ERROR("Exception in DataRequester request(): request id:%ld , request context id:%ld : %s",
                        requestAndPromise.mRequest->mRequestId,
                        requestAndPromise.mRequest->getContextPhaseParams().value().getReqId(), err.what());
                    requestAndPromise.mPromise->set_exception(std::current_exception());
                }
            }
        }
    }

    std::unique_ptr<DataReceiver> mReceiver;
    int mDeviceId{-1};

    std::vector<std::future<void>> mRequestFutures;
    std::unordered_map<std::string, std::unique_ptr<AsyncResource>> mInstanceToAsyncResource;
};

DataResponder::DataResponder(std::unique_ptr<DataSender> sender)
    : mImpl{std::make_unique<Impl>(std::move(sender))}
{
}

std::future<void> DataResponder::respondAndSendAsync(LlmRequest& llmRequest) const
{
    return mImpl->respondAndSendAsync(llmRequest);
}

executor::kv_cache::CommState const& DataResponder::getCommState() const
{
    return mImpl->getCommState();
}

DataResponder::~DataResponder() = default;

DataRequester::DataRequester(std::unique_ptr<DataReceiver> receiver)
    : mImpl{std::make_unique<Impl>(std::move(receiver))}
{
}

std::future<void> DataRequester::requestAndReceiveAsync(LlmRequest& llmRequest) const
{
    return mImpl->requestAndReceiveAsyncMultiThreads(llmRequest);
}

DataRequester::~DataRequester() = default;

} // namespace tensorrt_llm::batch_manager
