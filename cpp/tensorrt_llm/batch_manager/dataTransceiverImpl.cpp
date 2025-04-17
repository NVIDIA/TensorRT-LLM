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

#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "nixl.h"
#include "nixl_descriptors.h" // For nixlBasicDesc
#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/mlaCacheFormatter.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/connection.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::batch_manager
{

DataSenderImpl::DataSenderImpl(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::ConnectionManager* nixlManager, executor::kv_cache::CacheState selfCacheState,
    SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter)
    : mManager{manager}
    , mNixlManager{nixlManager}
    , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
    , mFormatter(std::move(formatter))
    , mBufferManager{std::make_shared<runtime::CudaStream>()}
{
    TLLM_CHECK(mManager);
    TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);

    // Register pre-allocated buffers with NIXL manager
    auto* nixlManager1 = dynamic_cast<executor::kv_cache::NixlConnectionManager*>(mNixlManager);
    if (nixlManager1)
    {
        auto* cacheFormatter = dynamic_cast<kv_cache_manager::CacheFormatter*>(mFormatter.get());
        auto* mlaCacheFormatter = dynamic_cast<kv_cache_manager::MLACacheFormatter*>(mFormatter.get());
        TLLM_CHECK(cacheFormatter != nullptr || mlaCacheFormatter != nullptr);

        std::vector<executor::kv_cache::NixlConnectionManager::BufferInfo> buffers;
        std::vector<runtime::ITensor::SharedPtr> preAllocatedSendBuffers;
        if (cacheFormatter != nullptr)
        {
            preAllocatedSendBuffers = cacheFormatter->getPreAllocatedSendBuffers();
        }
        else
        {
            preAllocatedSendBuffers = mlaCacheFormatter->getPreAllocatedSendBuffers();
        }
        buffers.reserve(preAllocatedSendBuffers.size());

        // Register send buffers
        for (auto const& buffer : preAllocatedSendBuffers)
        {
            cudaPointerAttributes attributes;
            cudaError_t err = cudaPointerGetAttributes(&attributes, buffer->data());
            if (err != cudaSuccess)
            {
                TLLM_LOG_ERROR("Failed to get CUDA pointer attributes: %s", cudaGetErrorString(err));
                continue;
            }

            buffers.push_back({buffer->data(), buffer->getSizeInBytes(), attributes.device});
        }

        if (!buffers.empty())
        {
            nixlManager1->registerBuffers(buffers);
        }
    }
}

[[nodiscard]] RequestInfo DataSenderImpl::recvRequestInfo()
{
    using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
    Id id;
    tensorrt_llm::executor::kv_cache::Connection const* connection = nullptr;
    std::string serializedInfo;
    std::istringstream iss;
    RequestInfo info = RequestInfo::deserialize(iss);
    auto requestId = info.getRequestId();

    if (mNixlManager) // if (common::getEnvUseNIXLKvCache())
    {
        std::string bufferDescStr;
        auto* nixlManager1 = dynamic_cast<executor::kv_cache::NixlConnectionManager*>(mNixlManager);
        connection = nixlManager1->recvRequestInfo(serializedInfo, bufferDescStr);
        iss = std::istringstream(serializedInfo);
        info = RequestInfo::deserialize(iss);
        requestId = info.getRequestId();
        auto bufferDesc = nixlBasicDesc(bufferDescStr);
        TLLM_LOG_DEBUG(
            "NIXL: Received request info %d from rank %d and buffer descriptor (addr: %p, size: %zu, devid: %d)",
            requestId, connection->getRank(), (void*) bufferDesc.addr, bufferDesc.len, bufferDesc.devId);
        // Receive buffer descriptor
        auto tag = REQUEST_INFO_TAG(requestId);
        nixlManager1->setRecvBufferInfo(connection->getRank(), tag, bufferDescStr);
    }
    else
    {
        connection = mManager->recvConnect(DataContext{kID_TAG}, &id, sizeof(id));
        TLLM_CHECK(id == Id::REQUEST_SEND);
        std::uint64_t infoSize{0};
        connection->recv(executor::kv_cache::DataContext{kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
        serializedInfo.resize(infoSize);
        connection->recv(executor::kv_cache::DataContext{kINFO_TAG}, serializedInfo.data(), infoSize);
        iss = std::istringstream(serializedInfo);
        info = RequestInfo::deserialize(iss);
        requestId = info.getRequestId();
    }

    TLLM_CHECK_WITH_INFO(
        mFormatter->inquireSupport(mSelfState.getCacheState().value(), info.getTransState().getCacheState().value()),
        "Disagg server does not currently support these cacheState.");
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

void DataSenderImpl::sendSync(LlmRequest const& llmRequest)
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
    mFormatter->formatOutput(llmRequest, std::move(connections), mSelfState.getCacheState().value(),
        mSelfState.getCommState().value().getSelfIdx(), dataTransceiverState.getCacheState().value(), mBufferManager);
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
    auto it = mRequestToComms.find(requestId);
    TLLM_CHECK(it != mRequestToComms.end());
    return it->second.size();
}

void DataSenderImpl::release(LlmRequest::RequestIdType requestId)
{
    auto it = mRequestToComms.find(requestId);
    TLLM_CHECK(it != mRequestToComms.end());
    std::unique_lock<std::mutex> lk(mMtxForMap);
    mRequestToComms.erase(it);
}

DataReceiverImpl::DataReceiverImpl(executor::kv_cache::ConnectionManager* manager,
    executor::kv_cache::ConnectionManager* nixlManager, executor::kv_cache::CacheState selfCacheState,
    SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter)
    : mManager{manager}
    , mNixlManager{nixlManager}
    , mSelfState{std::move(selfCacheState), executor::kv_cache::CommState{manager->getCommState()}}
    , mFormatter(std::move(formatter))
{
    TLLM_CHECK(mManager);
    TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
    TLLM_CHECK(mFormatter);

    // Register pre-allocated buffers with NIXL manager
    auto* nixlManager1 = dynamic_cast<executor::kv_cache::NixlConnectionManager*>(mNixlManager);
    if (nixlManager1)
    {
        auto* cacheFormatter = dynamic_cast<kv_cache_manager::CacheFormatter*>(mFormatter.get());
        auto* mlaCacheFormatter = dynamic_cast<kv_cache_manager::MLACacheFormatter*>(mFormatter.get());
        TLLM_CHECK(cacheFormatter != nullptr || mlaCacheFormatter != nullptr);
        std::vector<runtime::ITensor::SharedPtr> preAllocatedRecvBuffers;
        if (cacheFormatter)
        {
            preAllocatedRecvBuffers = cacheFormatter->getPreAllocatedRecvBuffers();
        }
        else
        {
            preAllocatedRecvBuffers = mlaCacheFormatter->getPreAllocatedRecvBuffers();
        }
        std::vector<executor::kv_cache::NixlConnectionManager::BufferInfo> buffers;

        for (auto const& buffer : preAllocatedRecvBuffers)
        {
            // Get the CUDA device ID where the buffer was allocated
            cudaPointerAttributes attributes;
            cudaError_t err = cudaPointerGetAttributes(&attributes, buffer->data());
            if (err != cudaSuccess)
            {
                TLLM_LOG_ERROR("Failed to get CUDA pointer attributes: %s", cudaGetErrorString(err));
                continue;
            }

            buffers.push_back({buffer->data(), buffer->getSizeInBytes(), attributes.device});
        }

        if (!buffers.empty())
        {
            nixlManager1->registerBuffers(buffers);
        }
    }
}

void DataReceiverImpl::sendRequestInfo(LlmRequest const& llmRequest)
{
    uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& commState = contextState.getCommState().value();
    auto const& destCacheState = contextState.getCacheState().value();
    TLLM_CHECK_WITH_INFO(mFormatter->inquireSupport(mSelfState.getCacheState().value(), destCacheState),
        "Disagg server does not currently support these cacheState.");

    RequestInfo requestInfo(requestId, mSelfState);

    // TODO: remove IOFormatter and make CacheFormatter new base class
    auto* cacheFormatter = dynamic_cast<kv_cache_manager::CacheFormatter const*>(mFormatter.get());
    auto* mlaCacheFormatter = dynamic_cast<kv_cache_manager::MLACacheFormatter const*>(mFormatter.get());
    TLLM_CHECK(cacheFormatter != nullptr || mlaCacheFormatter != nullptr);
    tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager* cacheManager;
    if (cacheFormatter != nullptr)
    {
        cacheManager = cacheFormatter->getCacheManager();
    }
    else
    {
        cacheManager = mlaCacheFormatter->getCacheManager();
    }

    auto blockRange
        = kv_cache_manager::BlockRange(*cacheManager, cacheManager->getNewlyAllocatedBlockIds(llmRequest.mRequestId));
    requestInfo = RequestInfo(requestId, blockRange.getBlockHashes(), mSelfState);

    std::vector<tensorrt_llm::executor::kv_cache::Connection const*> connections, chosenConnections;

    for (auto index : mFormatter->getCounterparts(
             mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
    {
        auto const* connection = mManager->getConnections(commState).at(index);
        if (mNixlManager)
        {
            int connectionRank = connection->getRank();
            Id id{Id::REQUEST_SEND};
            connection = mNixlManager->recvConnect(
                tensorrt_llm::executor::kv_cache::DataContext{connectionRank}, &id, sizeof(id));
        }
        connections.emplace_back(connection);
    }
    // TODO: fix later with TRT-LLM team
    // if (mlaCacheFormatter)
    //{
    //     chosenConnections = mlaCacheFormatter->pickRecvConnections(connections, mSelfState.getCacheState().value(),
    //          mSelfState.getCommState().value().getSelfIdx(), destCacheState);
    // }
    // else
    {
        chosenConnections = connections;
    }

    for (auto connection : chosenConnections)
    {
        sendRequestInfo(connection, requestInfo);
    }
}

void DataReceiverImpl::receiveSync(LlmRequest const& llmRequest)
{
    auto const& contextState = llmRequest.getDataTransceiverState();
    auto const& commState = contextState.getCommState().value();
    auto const& destCacheState = contextState.getCacheState().value();
    std::vector<tensorrt_llm::executor::kv_cache::Connection const*> connections;
    for (auto index : mFormatter->getCounterparts(
             mSelfState.getCacheState().value(), mSelfState.getCommState().value().getSelfIdx(), destCacheState))
    {
        auto const* connection = mManager->getConnections(commState).at(index);
        if (mNixlManager)
        {
            int connectionRank = connection->getRank();
            Id id{Id::REQUEST_SEND};
            connection = mNixlManager->recvConnect(
                tensorrt_llm::executor::kv_cache::DataContext{connectionRank}, &id, sizeof(id));
        }
        connections.emplace_back(connection);
    }
    auto const& resource = getReceiveCacheResource(llmRequest);
    mFormatter->formatInput(llmRequest, std::move(connections), mSelfState.getCacheState().value(),
        mSelfState.getCommState().value().getSelfIdx(), destCacheState, resource->mBufferManager);
}

void DataReceiverImpl::sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info)
{
    std::ostringstream oss;
    RequestInfo::serialize(info, oss);
    auto const& serializedInfo = oss.str();
    std::size_t const infoSize = serializedInfo.size();
    Id id{Id::REQUEST_SEND};

    if (mNixlManager) // if (common::getEnvUseNIXLKvCache())
    {
        auto* nixlManager1 = dynamic_cast<executor::kv_cache::NixlConnectionManager*>(mNixlManager);
        int rank = connection->getRank();
        auto* cacheFormatter = dynamic_cast<kv_cache_manager::CacheFormatter const*>(mFormatter.get());
        auto* mlaCacheFormatter = dynamic_cast<kv_cache_manager::MLACacheFormatter const*>(mFormatter.get());
        TLLM_CHECK(cacheFormatter != nullptr || mlaCacheFormatter != nullptr);
        std::string buffertag = GET_BUFFER_TAG(info.getRequestId(), rank);
        runtime::ITensor::SharedPtr recvBuffer;
        if (cacheFormatter != nullptr)
        {
            recvBuffer = cacheFormatter->getPreAllocatedRecvBuffer(buffertag);
        }
        else
        {
            recvBuffer = mlaCacheFormatter->getPreAllocatedRecvBuffer(buffertag);
        }
        TLLM_CHECK(recvBuffer != nullptr);
        nixlManager1->sendRequestInfo(rank, serializedInfo, recvBuffer);
    }
    else
    {
        connection->send(executor::kv_cache::DataContext{kID_TAG}, &id, sizeof(id));
        connection->send(executor::kv_cache::DataContext{kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
        connection->send(executor::kv_cache::DataContext{kINFO_TAG}, serializedInfo.data(), serializedInfo.size());
    }
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
