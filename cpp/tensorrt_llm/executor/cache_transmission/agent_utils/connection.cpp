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

#include "connection.h"
#include "tensorrt_llm/common/envUtils.h"
#include <string>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

std::string genUniqueAgentName()
{
    static std::atomic<uint64_t> counter{0};

    // hostname+pid+counter++
    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    auto pid = static_cast<uint64_t>(::getpid());
    return std::string(hostname) + "_" + std::to_string(pid) + "_" + std::to_string(counter++);
}

AgentConnection::AgentConnection(
    std::string mAgentName, std::string mRemoteAgentName, AgentConnectionManager* mAgentConnectionManager)
    : mAgentName(mAgentName)
    , mRemoteAgentName(mRemoteAgentName)
    , mAgentConnectionManager(mAgentConnectionManager)
    , mCacheTransBufferManager(mAgentConnectionManager->getCacheTransBufferManager())
    , mNeedSendMetadata(true)
{
    TLLM_CHECK(mAgentConnectionManager != nullptr);
    TLLM_CHECK(mCacheTransBufferManager != nullptr);
}

std::optional<size_t> AgentConnection::getCacheBufferId() const
{
    return mCacheBufferId;
}

void MemoryDesc::serialize(MemoryDesc const& memoryDesc, std::ostream& os)
{
    namespace su = executor::serialize_utils;
    su::serialize(memoryDesc.mAddr, os);
    su::serialize(memoryDesc.mLen, os);
    su::serialize(memoryDesc.mDeviceId, os);
}

MemoryDesc MemoryDesc::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto addr = su::deserialize<decltype(mAddr)>(is);
    auto len = su::deserialize<decltype(mLen)>(is);
    auto deviceId = su::deserialize<decltype(mDeviceId)>(is);
    return MemoryDesc{addr, len, deviceId};
}

size_t MemoryDesc::serializedSize(MemoryDesc const& memoryDesc)
{
    namespace su = executor::serialize_utils;
    return su::serializedSize(memoryDesc.mAddr) + su::serializedSize(memoryDesc.mLen)
        + su::serializedSize(memoryDesc.mDeviceId);
}

void AgentConnection::send(DataContext const& ctx, void const* data, size_t size) const
{

    MemoryDesc srcDesc{
        reinterpret_cast<uintptr_t>(data), size, static_cast<uint32_t>(mAgentConnectionManager->getDeviceId())};
    MemoryDescs srcDescs{MemoryType::kVRAM, {srcDesc}};
    auto dstBaseDesc = mSenderState.mReceiverBufferDesc;
    MemoryDesc dstDesc{dstBaseDesc.getAddr() + (mSenderState.validSegmentIdx * size), size, dstBaseDesc.getDeviceId()};
    TLLM_LOG_DEBUG(
        "send dstDesc: %p, size: %ld ,validSegmentIdx: %ld", dstDesc.getAddr(), size, mSenderState.validSegmentIdx);
    MemoryDescs dstDescs{MemoryType::kVRAM, {dstDesc}};
    TransferRequest request{TransferOp::kWRITE, srcDescs, dstDescs, mRemoteAgentName};
    auto status = mAgentConnectionManager->getAgent()->submitTransferRequests(request);
    NotificationSyncInfo syncInfo{mRemoteAgentName, ctx};
    NotificationInfo notificationInfo{syncInfo};
    std::stringstream ss;
    NotificationInfo::serialize(notificationInfo, ss);
    status->wait();
    // TODO: there is a bug in request_with_notify https://github.com/ai-dynamo/nixl/pull/252
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
}

void AgentConnection::recv(DataContext const& ctx, void* data, size_t size) const
{

    NotificationSyncInfo syncInfo{mAgentName, ctx};
    mAgentConnectionManager->waitForSyncInfo(mRemoteAgentName, syncInfo);
}

void AgentConnection::sendRequestAndBufferInfo(
    batch_manager::RequestInfo& requestInfo, std::optional<size_t> cacheBufferId, int validConnectionIdx)
{
    TLLM_CHECK(!common::getEnvTryZCopyForKVCacheTransfer());

    TLLM_CHECK(cacheBufferId.has_value());
    auto preAllocateBuffer = mCacheTransBufferManager->getRecvBuffer(cacheBufferId.value());
    // memory Desp , validSegmentIdx send
    mCacheBufferId = cacheBufferId;
    // TODO: deviceID;
    int deviceId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    TLLM_CHECK(deviceId != -1);
    TLLM_CHECK(deviceId == mAgentConnectionManager->getDeviceId());
    MemoryDesc bufferDesc(
        reinterpret_cast<uintptr_t>(preAllocateBuffer->data()), preAllocateBuffer->getSize(), deviceId);
    std::string address = mAgentConnectionManager->getAgent()->getConnectionInfo();
    std::optional<std::string> metadataOpt = std::nullopt;
    if (mNeedSendMetadata)
    {
        auto metadata = mAgentConnectionManager->getAgent()->getLocalAgentDesc().getBackendAgentDesc();
        metadataOpt = metadata;
        mNeedSendMetadata = false;
    }

    RequestAndBufferInfo requestAndBufferInfo{
        mAgentName, address, requestInfo, bufferDesc, metadataOpt, validConnectionIdx};
    std::stringstream ss;
    NotificationInfo notificationInfo{requestAndBufferInfo};
    NotificationInfo::serialize(notificationInfo, ss);
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
}

void AgentConnection::setSenderState(MemoryDesc mReceiverBufferDesc, int validSegmentIdx)
{
    mSenderState.mReceiverBufferDesc = mReceiverBufferDesc;
    mSenderState.validSegmentIdx = validSegmentIdx;
}

void AgentConnection::setHasLoadRemoteAgent(bool hasLoadRemoteAgent)
{

    mHasLoadRemoteAgent = hasLoadRemoteAgent;
}

bool AgentConnection::hasLoadRemoteAgent() const
{
    return mHasLoadRemoteAgent;
}

AgentConnectionManager::AgentConnectionManager(
    batch_manager::kv_cache_manager::CacheTransBufferManager* cacheTransBufferManager)
    : mRegMemDescs(MemoryType::kVRAM, {})
{
    TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    TLLM_CHECK(mDeviceId != -1);

    mAgentName = genUniqueAgentName();
    // Create Agent
    BaseAgentConfig config{mAgentName, true};
    m_Agent = makeTransferAgent("nixl", &config);
    mCacheTransBufferManager = cacheTransBufferManager;
    auto recvBufferCount = mCacheTransBufferManager->getRecvBufferCount();
    auto sendBufferCount = mCacheTransBufferManager->getSendBufferCount();
    std::vector<MemoryDesc> MemDescs;
    for (size_t i = 0; i < recvBufferCount; i++)
    {
        auto recvBuffer = mCacheTransBufferManager->getRecvBuffer(i);
        MemDescs.emplace_back(recvBuffer->data(), recvBuffer->getSizeInBytes(), mDeviceId);
    }
    for (size_t i = 0; i < sendBufferCount; i++)
    {
        auto sendBuffer = mCacheTransBufferManager->getSendBuffer(i);
        MemDescs.emplace_back(sendBuffer->data(), sendBuffer->getSizeInBytes(), mDeviceId);
    }
    mRegMemDescs = MemoryDescs{MemoryType::kVRAM, MemDescs};
    m_Agent->registerMemory(mRegMemDescs);

    AgentState localAgentState{mAgentName, m_Agent->getConnectionInfo()};
    std::vector<AgentState> agentStates(mpi::MpiComm::session().getSize());
    if (mpi::MpiComm::session().getSize() > 1)
    {

        mpi::MpiComm::session().barrier();
        namespace su = executor::serialize_utils;

        std::ostringstream oStream;
        su::serialize(localAgentState, oStream);
        auto str = oStream.str();
        std::vector<char> buffer(str.begin(), str.end());
        std::vector<SizeType32> sizeofBuffer(mpi::MpiComm::session().getSize());
        SizeType32 bufferSize = buffer.size();
        mpi::MpiComm::session().allgather(&bufferSize, sizeofBuffer.data(), 1, mpi::MpiType::kINT32);
        SizeType32 recvBufferSize = std::accumulate(sizeofBuffer.begin(), sizeofBuffer.end(), 0);
        std::vector<char> recvBuffer(recvBufferSize);
        std::vector<int> displs(mpi::MpiComm::session().getSize());
        for (int r = 0; r < mpi::MpiComm::session().getSize(); r++)
        {
            displs[r] = (r == 0) ? 0 : (displs[r - 1] + sizeofBuffer[r - 1]);
        }
        mpi::MpiComm::session().allgatherv(buffer.data(), bufferSize, mpi::MpiType::kCHAR, recvBuffer.data(),
            sizeofBuffer, displs, mpi::MpiType::kCHAR);

        // deserialize
        for (int i = 0; i < mpi::MpiComm::session().getSize(); i++)
        {
            std::vector<char> serBuffer(
                recvBuffer.begin() + displs[i], recvBuffer.begin() + (displs[i] + sizeofBuffer[i]));
            su::VectorWrapBuf<char> strbuf(serBuffer);
            std::istream is(&strbuf);
            agentStates[i] = su::deserialize<executor::kv_cache::AgentState>(is);
            TLLM_LOG_DEBUG(
                mpi::MpiComm::world().getRank(), " recv  agentStates[%d]: %s", i, agentStates[i].toString().c_str());
        }
    }
    else
    {
        agentStates[0] = localAgentState;
    }
    mCommState = CommState(agentStates, mpi::MpiComm::session().getRank());
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        " ***** AgentConnectionManager::AgentConnectionManager    mCommState: %s", mCommState.toString().c_str());
}

AgentConnection const* AgentConnectionManager::recvConnectionAndRequestInfo(batch_manager::RequestInfo& requestInfo)
{
    // recv remoteAgentDesc, and bufferDesc , and validSegmentIdx ,

    while (true)
    {

        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            auto it2 = notifs.begin();
            while (it2 != notifs.end())
            {
                std::stringstream ss(*it2);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
                {
                    auto requestAndBufferInfo = std::get<RequestAndBufferInfo>(notificationInfo.mInfo);

                    erase = true;
                    requestInfo = requestAndBufferInfo.mRequestInfo;
                    auto address = requestAndBufferInfo.mAddress;
                    auto bufferDesc = requestAndBufferInfo.mBufferDesc;
                    auto metadataOpt = requestAndBufferInfo.mMetadata;
                    auto validConnectionIdx = requestAndBufferInfo.mValidConnectionIdx;
                    auto remoteAgentName = requestAndBufferInfo.mAgentName;
                    TLLM_LOG_DEBUG(" recv Address:%s", address.c_str());
                    auto connection = connect(remoteAgentName, address, metadataOpt, true);
                    connection->setSenderState(bufferDesc, validConnectionIdx);
                    it2 = notifs.erase(it2);
                    if (notifs.empty())
                    {
                        it = mUnhandledNotifications.erase(it);
                    }
                    return connection;
                }

                if (!erase)
                {
                    it2++;
                }
            }
            if (notifs.empty())
            {
                it = mUnhandledNotifications.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
    return nullptr;
}

void AgentConnectionManager::updateUnhandledNotifications()
{

    auto notif_map = m_Agent->getNotifiedSyncMessages();
    std::lock_guard<std::mutex> lock(mNotificationMutex);

    // Merge new notifications with existing ones
    for (auto const& [agent, notifs] : notif_map)
    {
        auto& existing_notifs = mUnhandledNotifications[agent];
        existing_notifs.insert(
            existing_notifs.end(), std::make_move_iterator(notifs.begin()), std::make_move_iterator(notifs.end()));
    }
}

[[nodiscard]] std::vector<Connection const*> AgentConnectionManager::getConnections(CommState const& state)
{
    //  agentDesc +ip
    // get metaData from ip;
    TLLM_CHECK(state.isAgentState());
    // TODO:  AgentCommState
    auto ret = std::vector<Connection const*>();
    for (auto&& agentState : state.getAgentState())
    {
        std::string agentName = agentState.mAgentName;
        std::string connectionInfo = agentState.mConnectionInfo;
        ret.emplace_back(connect(agentName, connectionInfo));
    }
    return ret;
}

BaseTransferAgent* AgentConnectionManager::getAgent() const
{
    return m_Agent.get();
}

batch_manager::kv_cache_manager::CacheTransBufferManager* AgentConnectionManager::getCacheTransBufferManager()
{
    return mCacheTransBufferManager;
}

AgentConnection* AgentConnectionManager::connect(std::string const& remoteAgentName, std::string const& connecitonInfo,
    std::optional<std::string> metadata, bool isSender)
{

    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "mAgentName: %s connect to %s", mAgentName.c_str(), remoteAgentName.c_str());
    std::scoped_lock lock(mConnectionsMutex);
    auto it = mConnections.find(remoteAgentName);
    if (it != mConnections.end())
    {
        if (isSender)
        {
            if (!it->second->hasLoadRemoteAgent())
            {
                TLLM_CHECK_WITH_INFO(metadata.has_value(), "should get metadata for sender loadRemtoeAgent");
            }
        }
        if (!it->second->hasLoadRemoteAgent() && metadata.has_value())
        {
            m_Agent->invalidateRemoteAgent(remoteAgentName);
            it->second->setHasLoadRemoteAgent(true);
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "set has load remote agent to true");
            m_Agent->loadRemoteAgent(remoteAgentName, AgentDesc{metadata.value()});
        }
        return it->second.get();
    }
    bool hasLoadRemoteAgent = false;
    if (remoteAgentName != mAgentName)
    {
        if (metadata.has_value())
        {
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "mAgentName: %s connect to %s with loadRemoteAgent",
                mAgentName.c_str(), remoteAgentName.c_str());
            m_Agent->loadRemoteAgent(remoteAgentName, AgentDesc{metadata.value()});
            hasLoadRemoteAgent = true;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(!isSender, "Sender shouldn't call connectRemoteAgent");
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "mAgentName: %s connect to %s with connectRemoteAgent",
                mAgentName.c_str(), remoteAgentName.c_str());
            m_Agent->connectRemoteAgent(remoteAgentName, connecitonInfo);
        }
    }
    else
    {
        hasLoadRemoteAgent = true;
    }

    auto connection = std::make_shared<AgentConnection>(mAgentName, remoteAgentName, this);
    mConnections[remoteAgentName] = connection;
    connection->setHasLoadRemoteAgent(hasLoadRemoteAgent);
    return connection.get();
}

CommState const& AgentConnectionManager::getCommState() const
{

    return mCommState;
}

AgentConnection* AgentConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{

    TLLM_THROW("Not implemented");
    return nullptr;
}

int AgentConnectionManager::getDeviceId() const
{
    return mDeviceId;
}

void AgentConnectionManager::waitForSyncInfo(std::string const& remoteAgentName, NotificationSyncInfo const& syncInfo)
{
    while (true)
    {

        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            if (agent != remoteAgentName)
            {
                it++;
                continue;
            }
            auto it2 = notifs.begin();
            while (it2 != notifs.end())
            {
                std::stringstream ss(*it2);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
                {
                    auto notificationSyncInfo = std::get<NotificationSyncInfo>(notificationInfo.mInfo);
                    if (notificationSyncInfo.mContext.getTag() == syncInfo.mContext.getTag()
                        && notificationSyncInfo.mAgentName == syncInfo.mAgentName)
                    {
                        erase = true;
                        it2 = notifs.erase(it2);
                        if (notifs.empty())
                        {
                            it = mUnhandledNotifications.erase(it);
                        }
                        return;
                    }
                }
                if (!erase)
                {
                    it2++;
                }
            }
            if (notifs.empty())
            {
                it = mUnhandledNotifications.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
}

std::string const& AgentConnectionManager::getAgentName() const
{
    return mAgentName;
}

AgentConnectionManager::~AgentConnectionManager()
{
    // TODO: invalideRemoteAgent
    m_Agent->deregisterMemory(mRegMemDescs);
}
} // namespace tensorrt_llm::executor::kv_cache
