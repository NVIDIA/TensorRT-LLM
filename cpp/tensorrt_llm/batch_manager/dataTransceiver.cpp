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
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
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

void TransferSession::appendMeasure(double delay, double duration, size_t size)
{
    if (!mRecordMeasure)
    {
        return;
    }
    auto bandwidth = size * 8 / (duration / 1000) / 1e9; // byte, ms => Gbps
    mMeasures.emplace_back(Measure{delay, duration, bandwidth});
}

void TransferSession::exportMeasure(std::ofstream& outFile, bool isContext) const
{
    if (mMeasures.empty())
    {
        return;
    }
    // write header if not exist
    if (outFile.tellp() == 0)
    {
        outFile << "RequestID";
        for (size_t i = 0; i < mMeasures.size(); i++)
        {
            outFile << ",Delay(ms),Duration(ms),Bandwidth(Gbps)";
        }
        outFile << '\n';
    }
    // write measures
    TLLM_CHECK(isContext || mRequest->getContextPhaseParams().has_value());
    auto reqId = isContext ? mRequest->mRequestId : mRequest->getContextPhaseParams().value().getReqId();
    outFile << reqId;
    for (auto const& measure : mMeasures)
    {
        outFile << "," << measure.delay << "," << measure.duration << "," << measure.bandwidth;
    }
    outFile << '\n' << std::flush;
}

std::vector<size_t> const& RequestInfo::getBlockHashes() const noexcept
{
    return mBlockHashes;
}

using runtime::SizeType32;
using AgentConnectionManager = tensorrt_llm::executor::kv_cache::AgentConnectionManager;
using DataContext = tensorrt_llm::executor::kv_cache::DataContext;

static int32_t tagFromRequestId(LlmRequest::RequestIdType requestId)
{
    constexpr int32_t kDATA_TAG{43};
    return ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
}

namespace fs = std::filesystem;

static fs::path getTransferOutputPath(char const* tag)
{
    auto outputPath = common::getEnvKVCacheTransferOutputPath();
    if (!outputPath.empty())
    {
        auto rank = mpi::MpiComm::world().getRank();
        auto path = fs::path(outputPath);
        fs::create_directories(path);
        return path / ("rank_" + std::to_string(rank) + "_" + tag + ".csv");
    }
    return {};
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
        int asyncSendThreadNum = common::getEnvKVCacheSendMaxConcurrenceNum();
        for (int i = 0; i < asyncSendThreadNum; i++)
        {
            mAsyncSendFutures.emplace_back(
                std::async(std::launch::async, &Impl::handleAsyncSend, this, std::ref(mAsyncSendResource)));
        }
    }

    [[nodiscard]] std::future<void> sendAsync(LlmRequest& llmRequest)
    {
        std::promise<void> promise;
        auto future = promise.get_future();
        {
            {
                std::unique_lock lkResp(mSenderMutex);
                mReadyResponses.emplace(
                    llmRequest.mRequestId, Response{std::addressof(llmRequest), std::move(promise)});
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
        std::unique_lock<std::mutex> lk(mMtxForMap);
        auto it = mRequestToSession.find(requestId);
        TLLM_CHECK(it != mRequestToSession.end());
        if (!common::getEnvKVCacheTransferOutputPath().empty())
        {
            if (!mMeasuresFile.is_open())
            {
                auto outputPath = getTransferOutputPath("send");
                mMeasuresFile.open(outputPath);
                TLLM_CHECK_WITH_INFO(
                    mMeasuresFile.is_open(), "Failed to open transfer output file: %s", outputPath.string().c_str());
            }
            it->second.exportMeasure(mMeasuresFile, true);
        }
        mRequestToSession.erase(it);
    }

    [[nodiscard]] RequestInfo recvRequestInfo()
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        bool isAgent = agentConnectionManager != nullptr;

        auto agentRecvFun = [&](RequestInfo& requestInfo)
        {
            auto const* connection = agentConnectionManager->recvConnectionAndRequestInfo(requestInfo);
            return connection;
        };
        TransceiverTag::Id id;
        RequestInfo info;
        auto const* connection = isAgent ? agentRecvFun(info)
                                         : mManager->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id, sizeof(id));
        if (!isAgent)
        {
            TLLM_CHECK(id == TransceiverTag::Id::REQUEST_SEND);
            std::uint64_t infoSize{0};
            connection->recv(
                executor::kv_cache::DataContext{TransceiverTag::kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
            std::string serializedInfo;
            serializedInfo.resize(infoSize);
            connection->recv(
                executor::kv_cache::DataContext{TransceiverTag::kINFO_TAG}, serializedInfo.data(), infoSize);
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
                    DataContext{tagFromRequestId(requestId)}, mSelfState, info.getTransState(), mBufferManager, nullptr,
                    !common::getEnvKVCacheTransferOutputPath().empty());
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
        mFormatter->format(*session);
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
            sendAndRemoveResponse(resp.mRequest->mRequestId, std::move(resp));
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
            TLLM_LOG_ERROR("Exception in sendAndRemoveResponse: %s ", e.what());
            resp.mPromise.set_exception(std::current_exception());
        }
    }

    void asyncSendAndRemoveResponse(RequestIdType id, Response resp) noexcept
    {
        std::unique_lock lk(mAsyncSendResource.mMtxForQueue);
        mAsyncSendResource.mSendQueue.emplace_back(std::move(resp));
        mAsyncSendResource.mCVforQueue.notify_one();
    }

    void sendResponse(std::vector<size_t> const& blockHashes, std::map<RequestIdType, Response>::iterator it)
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

            asyncSendAndRemoveResponse(it->first, std::move(it->second));
            removeResponse(it);
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
                std::vector<size_t> blockHashes;
                if (!mReadyResponses.empty())
                {
                    auto const& requestInfo = recvRequestInfo();
                    auto reqId = requestInfo.getRequestId();
                    blockHashes = requestInfo.getBlockHashes();

                    mCurrentRequest = reqId;
                    if (mRemainSendCount.find(reqId) == mRemainSendCount.end())
                    {
                        mRemainSendCount[reqId] = getCounterpartsCount(reqId);
                    }
                }
                auto it = getCurrentResponse();
                if (it != mReadyResponses.end())
                {
                    sendResponse(blockHashes, it);
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
                    sendResponse(blockHashes, it);
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
    }

    void removeResponse(std::map<RequestIdType, Response>::iterator it)
    {
        {
            std::unique_lock lkResp(mSenderMutex);
            mReadyResponses.erase(it);
        }
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
        std::unique_lock lk(mSenderMutex);
        return mReadyResponses.find(getCurrentRequestId());
    }

private:
    std::optional<RequestIdType> mCurrentRequest;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mSenderMutex, mCondMutex;
    std::atomic<bool> mAnyReady{false}, mTerminate{false};
    std::condition_variable mSenderCv;
    std::future<void> mResponseFuture;
    std::unordered_map<LlmRequest::RequestIdType, int> mRemainSendCount;
    AsyncSendResource mAsyncSendResource;
    std::vector<std::future<void>> mAsyncSendFutures;
    int mDeviceId{-1};

    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, TransferSession> mRequestToSession;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<BaseCacheFormatter> mFormatter;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
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
        if (!common::getEnvKVCacheTransferOutputPath().empty())
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
        TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
            "Disagg server does not currently support these cacheState.");

        RequestInfo requestInfo(requestId, mSelfState);

        auto disableSelectiveCacheTransfer = common::getEnvDisableSelectiveCacheTransfer()
            || (mFormatter->getCacheManager()->getBlockManager().getNumPools() > 1);
        if (!disableSelectiveCacheTransfer)
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
            contextState, resource->mBufferManager, &llmRequest, !common::getEnvKVCacheTransferOutputPath().empty());
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
        connection->send(executor::kv_cache::DataContext{TransceiverTag::kID_TAG}, &id, sizeof(id));
        connection->send(executor::kv_cache::DataContext{TransceiverTag::kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
        connection->send(executor::kv_cache::DataContext{TransceiverTag::kINFO_TAG}, serializedInfo.data(), infoSize);
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

    int mDeviceId{-1};
    std::vector<std::future<void>> mRequestFutures;
    std::unordered_map<std::string, std::unique_ptr<AsyncResource>> mInstanceToAsyncResource;
    executor::kv_cache::ConnectionManager* mManager;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<BaseCacheFormatter> mFormatter;
    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
    std::mutex mMeasuresFileMutex;
};

void CacheSender::ImplDeleter::operator()(Impl* ptr)
{
    delete ptr;
}

void CacheReceiver::ImplDeleter::operator()(Impl* ptr)
{
    delete ptr;
}

CacheSender::CacheSender(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
    SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mImpl{std::unique_ptr<Impl, ImplDeleter>(new Impl(manager, selfCacheState, selfIndex, std::move(formatter)))}
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

void CacheSender::sendSync(LlmRequest const& llmRequest)
{
    mImpl->sendSync(llmRequest);
}

RequestInfo CacheSender::recvRequestInfo()
{
    return mImpl->recvRequestInfo();
}

CacheReceiver::CacheReceiver(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mImpl{std::unique_ptr<Impl, ImplDeleter>(new Impl(manager, selfCacheState, selfIndex, std::move(formatter)))}
{
}

std::future<void> CacheReceiver::receiveAsync(LlmRequest& llmRequest) const
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

} // namespace tensorrt_llm::batch_manager
