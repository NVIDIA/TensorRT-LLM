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

#include "dataTransceiverImpl.h"

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::batch_manager
{

static int32_t tagFromRequestId(LlmRequest::RequestIdType requestId)
{
    constexpr int32_t kDATA_TAG{43};
    return ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
}

DataSenderImpl::DataSenderImpl(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mManager{manager}
    , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
    , mFormatter(std::move(formatter))
    , mBufferManager{std::make_shared<runtime::CudaStream>()}
{
    TLLM_CHECK(mManager);
    TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
    mFormatter->markAsSender(true);
}

[[nodiscard]] RequestInfo DataSenderImpl::recvRequestInfo()
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
        auto it = mRequestToSession.find(requestId);
        if (it == mRequestToSession.end())
        {
            auto session = TransferSession(std::vector<Connection const*>(peerRelativeRanks.size(), nullptr),
                DataContext{tagFromRequestId(requestId)}, mSelfState, info.getTransState(), mBufferManager);
            it = mRequestToSession.emplace(requestId, std::move(session)).first;
        }
        it->second.setConnection(peerIdx, connection);
    }
    return info;
}

void DataSenderImpl::sendSync(LlmRequest const& llmRequest)
{
    auto it = mRequestToSession.find(llmRequest.mRequestId);
    TLLM_CHECK(it != mRequestToSession.end());
    auto& session = it->second;
    session.setLlmRequest(llmRequest);
    mFormatter->format(session);
}

[[nodiscard]] executor::kv_cache::CommState const& DataSenderImpl::getCommState() const
{
    return mSelfState.getCommState().value();
}

void DataSenderImpl::setCommState(executor::kv_cache::CommState commState)
{
    mSelfState.setCommState(std::move(commState));
}

[[nodiscard]] size_t DataSenderImpl::getCounterpartsCount(LlmRequest::RequestIdType requestId) const
{
    auto it = mRequestToSession.find(requestId);
    TLLM_CHECK(it != mRequestToSession.end());
    return it->second.getConnections().size();
}

void DataSenderImpl::release(LlmRequest::RequestIdType requestId)
{
    auto it = mRequestToSession.find(requestId);
    TLLM_CHECK(it != mRequestToSession.end());
    std::unique_lock<std::mutex> lk(mMtxForMap);
    mRequestToSession.erase(it);
}

DataReceiverImpl::DataReceiverImpl(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::CacheState selfCacheState, SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter)
    : mManager{manager}
    , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
    , mFormatter(std::move(formatter))
{
    TLLM_CHECK(mManager);
    TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
    TLLM_CHECK(mFormatter);
    mFormatter->markAsSender(false);
}

TransferSession DataReceiverImpl::sendRequestInfo(LlmRequest const& llmRequest)
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
        contextState, resource->mBufferManager, &llmRequest);
}

void DataReceiverImpl::receiveSync(TransferSession& session)
{
    mFormatter->unformat(session);
}

void DataReceiverImpl::sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info)
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

std::unique_ptr<DataReceiverImpl::ReceiveCacheResource> const& DataReceiverImpl::getReceiveCacheResource(
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

} // namespace tensorrt_llm::batch_manager
