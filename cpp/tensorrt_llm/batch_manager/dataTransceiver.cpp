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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <future>
#include <map>
#include <memory>
#include <unordered_map>

namespace tensorrt_llm::batch_manager
{

using kv_cache_manager::BlockRange;
using runtime::SizeType32;
using AgentConnectionManager = tensorrt_llm::executor::kv_cache::AgentConnectionManager;
using DataContext = tensorrt_llm::executor::kv_cache::DataContext;

static int32_t tagFromRequestId(LlmRequest::RequestIdType requestId)
{
    constexpr int32_t kDATA_TAG{43};
    return ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
}

struct ReceiveCacheResource
{
    runtime::BufferManager mBufferManager;
    runtime::CudaEvent mCudaEvent;

    ReceiveCacheResource(runtime::BufferManager&& bufferManager, runtime::CudaEvent&& cudaEvent)
        : mBufferManager(bufferManager)
        , mCudaEvent(std::move(cudaEvent))
    {
    }
};

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState)
    : mRequestId{requestId}
    , mTransState{std::move(transState)}
{
}

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, std::vector<size_t> allBlockHashes,
    executor::DataTransceiverState transState, std::vector<size_t> requestedBlockHashes)
    : mRequestId{requestId}
    , mAllBlockHashes{std::move(allBlockHashes)}
    , mRequestedBlockHashes{std::move(requestedBlockHashes)}
    , mTransState{std::move(transState)}
{
}

bool RequestInfo::operator==(RequestInfo const& rhs) const
{
    return mRequestId == rhs.mRequestId && mAllBlockHashes == rhs.mAllBlockHashes
        && mRequestedBlockHashes == rhs.mRequestedBlockHashes && mTransState == rhs.mTransState;
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
    su::serialize(requestInfo.mAllBlockHashes, os);
    su::serialize(requestInfo.mRequestedBlockHashes, os);
    su::serialize(requestInfo.mTransState, os);
}

RequestInfo RequestInfo::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto requestId = su::deserialize<decltype(mRequestId)>(is);
    auto allBlockHashes = su::deserialize<decltype(mAllBlockHashes)>(is);
    auto requestedBlockHashes = su::deserialize<decltype(mRequestedBlockHashes)>(is);
    auto transState = su::deserialize<decltype(mTransState)>(is);
    return RequestInfo{requestId, std::move(allBlockHashes), std::move(transState), std::move(requestedBlockHashes)};
}

std::size_t RequestInfo::serializedSize(RequestInfo const& requestInfo)
{
    namespace su = executor::serialize_utils;
    std::size_t totalSize = 0;
    totalSize += su::serializedSize(requestInfo.mRequestId);
    totalSize += su::serializedSize(requestInfo.mAllBlockHashes);
    totalSize += su::serializedSize(requestInfo.mRequestedBlockHashes);
    totalSize += su::serializedSize(requestInfo.mTransState);
    return totalSize;
}

class CacheSender::Impl
{
public:
    using RequestIdType = LlmRequest::RequestIdType;

    Impl(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
        : mManager{manager}
        , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
        , mFormatter{std::move(formatter)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
        mCurrentRequest = std::nullopt;
        mResponseFuture = std::async(std::launch::async, &Impl::response, this);
    }

    [[nodiscard]] std::future<void> sendAsync(LlmRequest& llmRequest)
    {
        std::promise<void> promise;
        auto future = promise.get_future();
        {
            {
                std::unique_lock lkResp(mSenderMutex);
                mReadyRequests.emplace(llmRequest.mRequestId, std::addressof(llmRequest));
                mReadyPromises.emplace(llmRequest.mRequestId, std::move(promise));
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

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId) const
    {
        auto it = mRequestToSession.find(requestId);
        TLLM_CHECK(it != mRequestToSession.end());
        return it->second.getConnections().size();
    }

    void release(LlmRequest::RequestIdType requestId)
    {
        auto it = mRequestToSession.find(requestId);
        TLLM_CHECK(it != mRequestToSession.end());
        std::unique_lock<std::mutex> lk(mMtxForMap);
        mRequestToSession.erase(it);
    }

    [[nodiscard]] RequestInfo recvRequestInfo()
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        bool isAgent = agentConnectionManager != nullptr;

        TransceiverTag::Id id;
        RequestInfo info;
        auto const* connection = isAgent ? agentConnectionManager->recvConnectionAndRequestInfo(info)
                                         : mManager->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id, sizeof(id));
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
        TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(
                                 mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
            "Disagg server does not currently support these cacheState, please check the cacheState of the context and "
            "gen "
            "executors");
        auto peerRelativeRanks = executor::kv_cache::targetIRanks(info.getTransState().getCacheState().value(),
            mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx())
                                     .mIRanks;
        int peerIdx = std::distance(peerRelativeRanks.begin(),
            std::find(
                peerRelativeRanks.begin(), peerRelativeRanks.end(), info.getTransState().getCommState()->getSelfIdx()));
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            if (it == mRequestToSession.end())
            {
                auto session = TransferSession(std::vector<Connection const*>(peerRelativeRanks.size(), nullptr),
                    DataContext{tagFromRequestId(requestId)}, mSelfState, info.getTransState(), mBufferManager,
                    info.getAllBlockHashes(), info.getRequestedBlockHashes());
                it = mRequestToSession.emplace(requestId, std::move(session)).first;
            }
            it->second.setConnection(peerIdx, connection);
        }
        return info;
    }

    void sendSync(LlmRequest const& llmRequest)
    {
        auto it = mRequestToSession.find(llmRequest.mRequestId);
        TLLM_CHECK(it != mRequestToSession.end());
        auto& session = it->second;
        session.setLlmRequest(llmRequest);
        mFormatter->format(session);
    }

    ~Impl()
    {
        terminate();
    }

private:
    void sendAndRemoveResponse(RequestIdType id) noexcept
    {
        LlmRequest* request = nullptr;
        std::promise<void> promise;

        // Extract request and promise
        {
            std::unique_lock lkResp(mSenderMutex);
            auto requestIt = mReadyRequests.find(id);
            auto promiseIt = mReadyPromises.find(id);

            if (requestIt != mReadyRequests.end() && promiseIt != mReadyPromises.end())
            {
                request = requestIt->second;
                promise = std::move(promiseIt->second);
                mReadyRequests.erase(requestIt);
                mReadyPromises.erase(promiseIt);
            }
        }

        if (request == nullptr)
        {
            return; // Request not found
        }

        try
        {
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            sendSync(*request);
            release(id);
            promise.set_value();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s ", e.what());
            promise.set_exception(std::current_exception());
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
                    mSenderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                }
                if (mTerminate)
                {
                    break;
                }
                if (!isSending() && !mReadyPromises.empty())
                {
                    auto const& requestInfo = recvRequestInfo();
                    auto reqId = requestInfo.getRequestId();

                    mCurrentRequest = reqId;
                    if (mRemainSendCount.find(reqId) == mRemainSendCount.end())
                    {
                        mRemainSendCount[reqId] = getCounterpartsCount(reqId);
                    }
                }
                auto reqId = mCurrentRequest.value();
                auto it = mReadyPromises.find(reqId);
                if (it != mReadyPromises.end())
                {
                    auto count = --mRemainSendCount[reqId];
                    TLLM_CHECK(count >= 0);
                    if (count == 0)
                    {
                        mRemainSendCount.erase(reqId);

                        if (common::getEnvParallelCacheSend())
                        {
                            // TODO: Use a thread pool and check for thread safety.
                            std::thread(&CacheSender::Impl::sendAndRemoveResponse, this, reqId).detach();
                        }
                        else
                        {
                            CacheSender::Impl::sendAndRemoveResponse(reqId);
                        }
                        removeResponse(reqId);
                    }
                    mCurrentRequest = std::nullopt;
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(!mCurrentRequest.has_value(),
                        "This executor does not have a prepared KV cache for request ID: %zu, and the "
                        "mReadyPromises size is: %zu. mpi rank :%d     ",
                        mCurrentRequest.value(), mReadyPromises.size(), mpi::MpiComm::world().getRank());
                    std::unique_lock lk(mCondMutex);
                    mSenderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                }
            }
        }
        catch (std::exception const& err)
        {
            TLLM_LOG_ERROR("Exception in CacheSender response: %s", err.what());
            for (auto& it : mReadyPromises)
            {
                it.second.set_exception(std::current_exception());
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
        mSenderCv.notify_all();
    }

    void removeResponse(RequestIdType id)
    {
        {
            std::unique_lock lkResp(mSenderMutex);
            mReadyRequests.erase(id);
            mReadyPromises.erase(id);
        }
        if (mReadyRequests.empty())
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

    [[nodiscard]] bool hasCurrentResponse()
    {
        std::unique_lock<std::mutex> lk(mSenderMutex);
        auto requestId = getCurrentRequestId();
        return mReadyRequests.find(requestId) != mReadyRequests.end()
            && mReadyPromises.find(requestId) != mReadyPromises.end();
    }

private:
    std::optional<RequestIdType> mCurrentRequest;
    std::map<RequestIdType, LlmRequest*> mReadyRequests;
    std::map<RequestIdType, std::promise<void>> mReadyPromises;
    std::mutex mSenderMutex, mCondMutex;
    std::atomic<bool> mAnyReady{false}, mTerminate{false};
    std::condition_variable mSenderCv;
    std::future<void> mResponseFuture;
    std::unordered_map<LlmRequest::RequestIdType, int> mRemainSendCount;
    int mDeviceId{-1};

    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, TransferSession> mRequestToSession;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<BaseCacheFormatter> mFormatter;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
};

class CacheReceiver::Impl
{
public:
    Impl(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
        : mManager{manager}
        , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
        , mFormatter{std::move(formatter)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    }

    [[nodiscard]] std::future<void> receiveAsync(LlmRequest& llmRequest)
    {
        // TODO: Modify the implementation here to avoid frequent thread creation.
        return std::async(std::launch::async, &CacheReceiver::Impl::requestSync, this, std::ref(llmRequest));
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
                auto requestFuture = std::async(std::launch::async, &CacheReceiver::Impl::request, this,
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

    void receiveSync(TransferSession& session)
    {
        mFormatter->unformat(session);
    }

    TransferSession sendRequestInfo(LlmRequest const& llmRequest)
    {
        uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& commState = contextState.getCommState().value();
        auto const& destCacheState = contextState.getCacheState().value();
        TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
            "Disagg server does not currently support these cacheState.");

        RequestInfo requestInfo(requestId, mSelfState);

        if (mFormatter->getCacheManager()->isEnableBlockReuse()
            && mFormatter->getCacheManager()->getBlockManager().getNumPools() == 1)
        {
            auto* cacheManager = mFormatter->getCacheManager();
            auto beam = 0;
            auto allBlockRange = BlockRange::fromAllBlockIds(*cacheManager, llmRequest.mRequestId, beam);
            auto requestedBlockRange = getBlockRangeForReceiving(cacheManager, llmRequest);

            auto const& uniqueTokens = llmRequest.getUniqueTokens(beam);
            auto blockedUniqueTokens = tensorrt_llm::batch_manager::kv_cache_manager::chopVectorIntoBlocks<UniqueToken>(
                uniqueTokens, uniqueTokens.size() - 1, mFormatter->getCacheManager()->getTokensPerBlock(), false);
            auto blockKeys
                = tensorrt_llm::batch_manager::kv_cache_manager::buildBlockKeys(blockedUniqueTokens, llmRequest);
            std::vector<size_t> allBlockHashes;
            for (auto const& blockKey : blockKeys)
            {
                allBlockHashes.push_back(tensorrt_llm::batch_manager::kv_cache_manager::BlockKeyHasher::hash(blockKey));
            }
            // Figure out the size difference for computing the requested block hashes
            std::vector<size_t> requestedBlockHashes;
            for (auto i = 0; i < allBlockHashes.size(); i++)
            {
                if (i >= requestedBlockRange.getBlockHashes().size())
                {
                    requestedBlockHashes.push_back(allBlockHashes[i]);
                }
            }

            requestInfo = RequestInfo(requestId, allBlockHashes, mSelfState, requestedBlockHashes);
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
        auto pickUpIdx = mFormatter->pickRecvConnections(counterParts.size(), mSelfState.getCacheState().value(),
            mSelfState.getCommState().value().getSelfIdx(), destCacheState);
        for (size_t i = 0; i < counterPartConnections.size(); i++)
        {
            auto const* connection = counterPartConnections[i];
            // if Manager is agentConnectionManager, then send request info to agent
            auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
            if (agentConnectionManager != nullptr)
            {
                // TODO: index -> validConnectionIdx conversion
                auto valideConnectionIdx = std::find(pickUpIdx.begin(), pickUpIdx.end(), i) - pickUpIdx.begin();
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
        auto const& resource = getReceiveCacheResource(llmRequest);
        return TransferSession(std::move(counterPartConnections), DataContext{tagFromRequestId(requestId)}, mSelfState,
            contextState, resource->mBufferManager, requestInfo.getAllBlockHashes(),
            requestInfo.getRequestedBlockHashes(), &llmRequest);
    }

    std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest)
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
        auto session = sendRequestInfo(llmRequest);
        receiveSync(session);
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
                    TLLM_LOG_ERROR("Exception in CacheReceiver request(): request id:%ld , request context id:%ld : %s",
                        requestAndPromise.mRequest->mRequestId,
                        requestAndPromise.mRequest->getContextPhaseParams().value().getReqId(), err.what());
                    requestAndPromise.mPromise->set_exception(std::current_exception());
                }
            }
        }
    }

    int mDeviceId{-1};
    std::vector<std::future<void>> mRequestFutures;
    std::unordered_map<std::string, std::unique_ptr<AsyncResource>> mInstanceToAsyncResource;
    executor::kv_cache::ConnectionManager* mManager;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<BaseCacheFormatter> mFormatter;
    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
    runtime::BufferManager mBufferManager;
};

CacheSender::CacheSender(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
    SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mImpl{std::make_unique<Impl>(manager, selfCacheState, selfIndex, std::move(formatter))}
{
}

std::future<void> CacheSender::sendAsync(LlmRequest& llmRequest) const
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

CacheReceiver::CacheReceiver(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mImpl{std::make_unique<Impl>(manager, selfCacheState, selfIndex, std::move(formatter))}
{
}

std::future<void> CacheReceiver::receiveAsync(LlmRequest& llmRequest) const
{
    return mImpl->requestAndReceiveAsyncMultiThreads(llmRequest);
}

CacheReceiver::~CacheReceiver() = default;

} // namespace tensorrt_llm::batch_manager
