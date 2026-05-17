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

#include "connection.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include <chrono>
#include <exception>
#include <list>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

namespace
{

std::string bufferKindsToString(std::vector<uint8_t> const& bufferKinds)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < bufferKinds.size(); i++)
    {
        if (i > 0)
        {
            os << ",";
        }
        os << static_cast<int>(bufferKinds[i]);
    }
    os << "]";
    return os.str();
}

std::string optionalBufferIdsToString(std::vector<std::optional<size_t>> const& cacheBufferIds)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < cacheBufferIds.size(); i++)
    {
        if (i > 0)
        {
            os << ",";
        }
        if (cacheBufferIds[i].has_value())
        {
            os << cacheBufferIds[i].value();
        }
        else
        {
            os << "null";
        }
    }
    os << "]";
    return os.str();
}

std::string memoryDescsToString(std::vector<MemoryDesc> const& bufferDescs)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < bufferDescs.size(); i++)
    {
        if (i > 0)
        {
            os << ",";
        }
        os << "{addr=" << bufferDescs[i].getAddr() << ",len=" << bufferDescs[i].getLen()
           << ",device=" << bufferDescs[i].getDeviceId() << "}";
    }
    os << "]";
    return os.str();
}

std::string offsetRatiosToString(std::vector<std::pair<size_t, size_t>> const& offsetRatios)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < offsetRatios.size(); i++)
    {
        if (i > 0)
        {
            os << ",";
        }
        os << "{" << offsetRatios[i].first << "/" << offsetRatios[i].second << "}";
    }
    os << "]";
    return os.str();
}

int requestInfoSelfIdx(batch_manager::RequestInfo const& requestInfo)
{
    auto const& commState = requestInfo.getTransState().getCommState();
    return commState.has_value() ? commState->getSelfIdx() : -1;
}

char const* notificationTypeName(NotificationInfo const& notificationInfo)
{
    if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
    {
        return "RequestAndBufferInfo";
    }
    if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
    {
        return "NotificationSyncInfo";
    }
    if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
    {
        return "ReadySignalInfo";
    }
    return "Unknown";
}

std::string notificationSummary(std::string const& serializedNotification)
{
    try
    {
        std::stringstream ss(serializedNotification);
        auto notificationInfo = NotificationInfo::deserialize(ss);
        std::ostringstream os;
        os << notificationTypeName(notificationInfo);
        if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
        {
            auto const& requestInfo = std::get<RequestAndBufferInfo>(notificationInfo.mInfo);
            os << "{agent=" << requestInfo.mAgentName << ",requestId=" << requestInfo.mRequestInfo.getRequestId()
               << ",peerSelfIdx=" << requestInfoSelfIdx(requestInfo.mRequestInfo)
               << ",connectionIdx=" << requestInfo.mValidConnectionIdx
               << ",bufferDescs=" << requestInfo.mBufferDescs.size()
               << ",bufferKinds=" << bufferKindsToString(requestInfo.mBufferKinds)
               << ",metadata=" << static_cast<int>(requestInfo.mMetadata.has_value()) << "}";
        }
        else if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
        {
            auto const& syncInfo = std::get<NotificationSyncInfo>(notificationInfo.mInfo);
            os << "{agent=" << syncInfo.mAgentName << ",tag=" << syncInfo.mContext.getTag() << "}";
        }
        else if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
        {
            auto const& readySignalInfo = std::get<ReadySignalInfo>(notificationInfo.mInfo);
            os << "{agent=" << readySignalInfo.mAgentName << ",tag=" << readySignalInfo.mContext.getTag()
               << ",isReady=" << static_cast<int>(readySignalInfo.mIsReady) << "}";
        }
        return os.str();
    }
    catch (std::exception const& e)
    {
        return std::string("deserialize-error{") + e.what() + "}";
    }
}

std::string pendingNotificationsSummary(
    std::unordered_map<std::string, std::list<std::string>> const& pendingNotifications, size_t maxEntries = 8)
{
    std::ostringstream os;
    size_t emitted = 0;
    for (auto const& [agent, notifications] : pendingNotifications)
    {
        for (auto const& notification : notifications)
        {
            if (emitted >= maxEntries)
            {
                os << "...";
                return os.str();
            }
            if (emitted > 0)
            {
                os << ";";
            }
            os << "from=" << agent << ":" << notificationSummary(notification);
            emitted++;
        }
    }
    return os.str();
}

} // namespace

std::string genUniqueAgentName()
{
    static std::atomic<uint64_t> counter{0};

    // Generate a per-process random suffix to disambiguate agents across containers
    // that may share the same hostname (--network host) and PID namespace.
    static uint64_t const sRandomSuffix = []()
    {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) | rd();
    }();

    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    auto pid = static_cast<uint64_t>(::getpid());
    return std::string(hostname) + "_" + std::to_string(pid) + "_" + std::to_string(sRandomSuffix) + "_"
        + std::to_string(counter++);
}

// NIXL connection is specific, and different from the UCX and mpi connection,
// since NIXL only support one-sided communication. gen send buffer metaData to
// context when it sending requestInfo, but don't send buffer offset, since
// unformmatter has not called yet, it didn't know the cacheSize and offset. We
// assume the recv_size is the same as the send_size. and compute the buffer
// offset according to  the layer num of the selfPPrank ,and previous PP rank's
// layer num, since the buffer size is ratio is equal to the layer num ratio
// except the VSWA case.

template <typename CacheStateT>
auto computeSendOffsetRatio(
    CacheStateT const& peerCacheState, int peerIdx, CacheStateT const& selfCacheState, int connectionIdx)
{
    auto peerTargetInfo = targetIRanks(selfCacheState, peerCacheState, peerIdx);
    size_t offsetLayer = 0;
    for (int i = 0; i < connectionIdx; i++)
    {
        offsetLayer += peerTargetInfo.getPeerPPDomainLayerNum(i);
    }

    size_t selfSendLayer = peerTargetInfo.getPeerPPDomainLayerNum(connectionIdx);
    return std::make_pair(offsetLayer, selfSendLayer);
}

AgentConnection::AgentConnection(
    std::string mAgentName, std::string mRemoteAgentName, AgentConnectionManager* mAgentConnectionManager)
    : mAgentName(mAgentName)
    , mRemoteAgentName(mRemoteAgentName)
    , mAgentConnectionManager(mAgentConnectionManager)
    , mCacheTransBufferManagers(mAgentConnectionManager->getCacheTransBufferManagers())
    , mNeedSendMetadata(true)
{
    TLLM_CHECK(mAgentConnectionManager != nullptr);
    TLLM_CHECK(!mCacheTransBufferManagers.empty());
}

MemoryDesc const& AgentConnection::SenderState::activeBufferDesc() const
{
    TLLM_CHECK(!mCacheReceiverBufferDescs.empty());
    TLLM_CHECK(mActiveBufferIdx < mCacheReceiverBufferDescs.size());
    return mCacheReceiverBufferDescs[mActiveBufferIdx];
}

std::pair<size_t, size_t> const& AgentConnection::SenderState::activeOffsetRatio() const
{
    TLLM_CHECK(!mOffsetRatios.empty());
    TLLM_CHECK(mActiveBufferIdx < mOffsetRatios.size());
    return mOffsetRatios[mActiveBufferIdx];
}

void AgentConnection::SenderState::setActiveBufferIdx(size_t bufferIdx) const
{
    TLLM_CHECK(bufferIdx < mCacheReceiverBufferDescs.size());
    mActiveBufferIdx = bufferIdx;
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
    auto const& dstBaseDesc = mSenderState.activeBufferDesc();
    auto const& offsetRatio = mSenderState.activeOffsetRatio();
    auto offset = size / offsetRatio.second * offsetRatio.first;
    MemoryDesc dstDesc{dstBaseDesc.getAddr() + offset, size, dstBaseDesc.getDeviceId()};
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection send begin: localAgent=%s remoteAgent=%s tag=%d size=%zu "
        "srcAddr=%zu srcDevice=%u dstBaseAddr=%zu dstAddr=%zu dstDevice=%u activeBufferIdx=%zu "
        "validSegmentIdx=%d offsetRatio=%zu/%zu",
        mAgentName.c_str(), mRemoteAgentName.c_str(), ctx.getTag(), size, srcDesc.getAddr(), srcDesc.getDeviceId(),
        dstBaseDesc.getAddr(), dstDesc.getAddr(), dstDesc.getDeviceId(), mSenderState.mActiveBufferIdx,
        mSenderState.validSegmentIdx, offsetRatio.first, offsetRatio.second);
    TLLM_LOG_DEBUG(
        "send dstDesc: %p, size: %ld ,validSegmentIdx: %ld", dstDesc.getAddr(), size, mSenderState.validSegmentIdx);
    MemoryDescs dstDescs{MemoryType::kVRAM, {dstDesc}};
    TransferRequest request{TransferOp::kWRITE, srcDescs, dstDescs, mRemoteAgentName};
    auto status = mAgentConnectionManager->getAgent()->submitTransferRequests(request);
    NotificationSyncInfo syncInfo{mRemoteAgentName, ctx};
    NotificationInfo notificationInfo{syncInfo};
    std::stringstream ss;
    NotificationInfo::serialize(notificationInfo, ss);
    TransferState transferState = status->wait();
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection send transfer wait done: localAgent=%s remoteAgent=%s tag=%d "
        "transferState=%d",
        mAgentName.c_str(), mRemoteAgentName.c_str(), ctx.getTag(), static_cast<int>(transferState));
    TLLM_CHECK_WITH_INFO(transferState == TransferState::kSUCCESS, "AgentConnection::send failed");
    // TODO: there is a bug in request_with_notify https://github.com/ai-dynamo/nixl/pull/252
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection send sync notified: localAgent=%s remoteAgent=%s tag=%d "
        "payloadBytes=%zu",
        mAgentName.c_str(), mRemoteAgentName.c_str(), ctx.getTag(), ss.str().size());
}

void AgentConnection::recv(DataContext const& ctx, void* data, size_t size) const
{

    NotificationSyncInfo syncInfo{mAgentName, ctx};
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection recv wait begin: localAgent=%s remoteAgent=%s expectedAgent=%s "
        "tag=%d size=%zu",
        mAgentName.c_str(), mRemoteAgentName.c_str(), syncInfo.mAgentName.c_str(), ctx.getTag(), size);
    mAgentConnectionManager->waitForSyncInfo(mRemoteAgentName, syncInfo, ctx.getTransferTerminate());
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection recv wait end: localAgent=%s remoteAgent=%s expectedAgent=%s "
        "tag=%d size=%zu",
        mAgentName.c_str(), mRemoteAgentName.c_str(), syncInfo.mAgentName.c_str(), ctx.getTag(), size);
}

void AgentConnection::sendRequestAndBufferInfo(batch_manager::RequestInfo& requestInfo,
    std::vector<std::optional<size_t>> const& cacheBufferIds, int connectionIdx)
{
    TLLM_CHECK(!common::getEnvTryZCopyForKVCacheTransfer());

    TLLM_CHECK(!cacheBufferIds.empty());
    TLLM_CHECK(cacheBufferIds.size() <= mCacheTransBufferManagers.size());

    auto const& allKinds = mAgentConnectionManager->getBufferKinds();
    std::vector<runtime::ITensor::SharedPtr> preAllocateBuffers;
    std::vector<MemoryDesc> bufferDescs;
    std::vector<std::optional<size_t>> activeCacheBufferIds;
    std::vector<uint8_t> activeKinds;

    for (size_t i = 0; i < cacheBufferIds.size(); i++)
    {
        if (!cacheBufferIds[i].has_value())
        {
            continue;
        }
        auto preAllocateBuffer = mCacheTransBufferManagers[i]->getRecvBuffer(cacheBufferIds[i].value());
        TLLM_CHECK(preAllocateBuffer != nullptr);
        preAllocateBuffers.push_back(preAllocateBuffer);
        activeCacheBufferIds.push_back(cacheBufferIds[i]);
        activeKinds.push_back(allKinds[i]);
    }
    TLLM_CHECK(!activeCacheBufferIds.empty());

    mCacheBufferIds = std::move(activeCacheBufferIds);
    mBufferKinds = activeKinds;

    int deviceId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    TLLM_CHECK(deviceId != -1);
    TLLM_CHECK(deviceId == mAgentConnectionManager->getDeviceId());
    for (auto const& buf : preAllocateBuffers)
    {
        bufferDescs.emplace_back(reinterpret_cast<uintptr_t>(buf->data()), buf->getSizeInBytes(), deviceId);
    }
    std::string address = mAgentConnectionManager->getAgent()->getLocalConnectionInfo();
    std::optional<std::string> metadataOpt = std::nullopt;
    if (mNeedSendMetadata)
    {
        auto metadata = mAgentConnectionManager->getAgent()->getLocalAgentDesc().getBackendAgentDesc();
        metadataOpt = metadata;
        mNeedSendMetadata = false;
    }

    RequestAndBufferInfo requestAndBufferInfo{
        mAgentName, address, requestInfo, bufferDescs, metadataOpt, connectionIdx, activeKinds};
    std::stringstream ss;
    NotificationInfo notificationInfo{requestAndBufferInfo};
    NotificationInfo::serialize(notificationInfo, ss);
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection sendRequestAndBufferInfo notify begin: localAgent=%s "
        "remoteAgent=%s requestId=%zu peerSelfIdx=%d connectionIdx=%d allBufferIds=%s activeBufferIds=%s "
        "activeKinds=%s bufferDescs=%s metadata=%d addressBytes=%zu payloadBytes=%zu",
        mAgentName.c_str(), mRemoteAgentName.c_str(), requestInfo.getRequestId(), requestInfoSelfIdx(requestInfo),
        connectionIdx, optionalBufferIdsToString(cacheBufferIds).c_str(),
        optionalBufferIdsToString(mCacheBufferIds).c_str(), bufferKindsToString(activeKinds).c_str(),
        memoryDescsToString(bufferDescs).c_str(), static_cast<int>(metadataOpt.has_value()), address.size(),
        ss.str().size());
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection sendRequestAndBufferInfo notify end: localAgent=%s "
        "remoteAgent=%s requestId=%zu connectionIdx=%d",
        mAgentName.c_str(), mRemoteAgentName.c_str(), requestInfo.getRequestId(), connectionIdx);
}

void AgentConnection::setSenderState(std::vector<MemoryDesc> cacheReceiverBufferDescs, int validSegmentIdx,
    std::vector<std::pair<size_t, size_t>> offsetRatios, std::vector<uint8_t> bufferKinds)
{
    TLLM_CHECK(!cacheReceiverBufferDescs.empty());
    TLLM_CHECK(offsetRatios.size() == cacheReceiverBufferDescs.size());
    TLLM_CHECK(bufferKinds.size() == cacheReceiverBufferDescs.size());
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection setSenderState: localAgent=%s remoteAgent=%s validSegmentIdx=%d "
        "bufferDescs=%s offsetRatios=%s bufferKinds=%s",
        mAgentName.c_str(), mRemoteAgentName.c_str(), validSegmentIdx,
        memoryDescsToString(cacheReceiverBufferDescs).c_str(), offsetRatiosToString(offsetRatios).c_str(),
        bufferKindsToString(bufferKinds).c_str());
    mSenderState.mCacheReceiverBufferDescs = std::move(cacheReceiverBufferDescs);
    mSenderState.validSegmentIdx = validSegmentIdx;
    mSenderState.mOffsetRatios = std::move(offsetRatios);
    mSenderState.setActiveBufferIdx(0);
    mBufferKinds = std::move(bufferKinds);
}

void AgentConnection::setHasLoadRemoteAgent(bool hasLoadRemoteAgent)
{
    mHasLoadRemoteAgent = hasLoadRemoteAgent;
}

bool AgentConnection::hasLoadRemoteAgent() const
{
    return mHasLoadRemoteAgent;
}

void AgentConnection::sendReadySignal(DataContext const& ctx, bool isReady) const
{
    ReadySignalInfo readySignalInfo{mRemoteAgentName, ctx, isReady};
    NotificationInfo notificationInfo{readySignalInfo};
    std::stringstream ss;
    NotificationInfo::serialize(notificationInfo, ss);
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection sendReadySignal notify begin: localAgent=%s remoteAgent=%s "
        "readyAgent=%s tag=%d isReady=%d payloadBytes=%zu",
        mAgentName.c_str(), mRemoteAgentName.c_str(), readySignalInfo.mAgentName.c_str(), ctx.getTag(),
        static_cast<int>(isReady), ss.str().size());
    mAgentConnectionManager->getAgent()->notifySyncMessage(mRemoteAgentName, ss.str());
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection sendReadySignal notify end: localAgent=%s remoteAgent=%s "
        "readyAgent=%s tag=%d isReady=%d",
        mAgentName.c_str(), mRemoteAgentName.c_str(), readySignalInfo.mAgentName.c_str(), ctx.getTag(),
        static_cast<int>(isReady));
}

bool AgentConnection::recvReadySignal(DataContext const& ctx) const
{
    ReadySignalInfo readySignalInfo{mAgentName, ctx, false};
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection recvReadySignal wait begin: localAgent=%s remoteAgent=%s "
        "expectedAgent=%s tag=%d",
        mAgentName.c_str(), mRemoteAgentName.c_str(), readySignalInfo.mAgentName.c_str(), ctx.getTag());
    mAgentConnectionManager->waitForReadySignal(mRemoteAgentName, readySignalInfo, ctx.getTransferTerminate());
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnection recvReadySignal wait end: localAgent=%s remoteAgent=%s "
        "expectedAgent=%s tag=%d isReady=%d",
        mAgentName.c_str(), mRemoteAgentName.c_str(), readySignalInfo.mAgentName.c_str(), ctx.getTag(),
        static_cast<int>(readySignalInfo.mIsReady));
    return readySignalInfo.mIsReady;
}

void AgentConnection::activateBuffer(uint8_t kind) const
{
    for (size_t i = 0; i < mBufferKinds.size(); i++)
    {
        if (mBufferKinds[i] == kind)
        {
            mSenderState.setActiveBufferIdx(i);
            return;
        }
    }
}

std::optional<size_t> AgentConnection::getPreAssignedBufferId(uint8_t kind) const
{
    for (size_t i = 0; i < mBufferKinds.size(); i++)
    {
        if (mBufferKinds[i] == kind && i < mCacheBufferIds.size())
        {
            return mCacheBufferIds[i];
        }
    }
    return std::nullopt;
}

AgentConnectionManager::AgentConnectionManager(
    std::vector<batch_manager::BaseTransBufferManager*> cacheTransBufferManagers, CacheState cacheState,
    std::string const& backendType, std::optional<CacheState::RnnCacheState> rnnCacheState)
    : mCacheState(std::move(cacheState))
    , mRnnCacheState(std::move(rnnCacheState))
    , mCacheTransBufferManagers(std::move(cacheTransBufferManagers))
    , mRegMemDescs(MemoryType::kVRAM, {})
{
    TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    TLLM_CHECK(mDeviceId != -1);

    mAgentName = genUniqueAgentName();
    // Create Agent
    BaseAgentConfig config{mAgentName, true, false, true};
    m_Agent = makeTransferAgent(backendType, &config);
    TLLM_CHECK(!mCacheTransBufferManagers.empty());
    mBufferKinds.reserve(mCacheTransBufferManagers.size());
    std::vector<MemoryDesc> memDescs;
    for (auto* cacheTransBufferManager : mCacheTransBufferManagers)
    {
        TLLM_CHECK(cacheTransBufferManager != nullptr);
        mBufferKinds.push_back(static_cast<uint8_t>(cacheTransBufferManager->getBufferKind()));
        auto recvBufferCount = cacheTransBufferManager->getRecvBufferCount();
        auto sendBufferCount = cacheTransBufferManager->getSendBufferCount();
        for (size_t i = 0; i < recvBufferCount; i++)
        {
            auto recvBuffer = cacheTransBufferManager->getRecvBuffer(i);
            memDescs.emplace_back(recvBuffer->data(), recvBuffer->getSizeInBytes(), mDeviceId);
        }
        for (size_t i = 0; i < sendBufferCount; i++)
        {
            auto sendBuffer = cacheTransBufferManager->getSendBuffer(i);
            memDescs.emplace_back(sendBuffer->data(), sendBuffer->getSizeInBytes(), mDeviceId);
        }
    }
    mRegMemDescs = MemoryDescs{MemoryType::kVRAM, memDescs};
    m_Agent->registerMemory(mRegMemDescs);
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnectionManager initialized local agent: agent=%s device=%d "
        "registeredBuffers=%zu backend=%s sessionRank=%d sessionSize=%d worldRank=%d",
        mAgentName.c_str(), mDeviceId, memDescs.size(), backendType.c_str(), mpi::MpiComm::session().getRank(),
        mpi::MpiComm::session().getSize(), mpi::MpiComm::world().getRank());

    AgentState localAgentState{mAgentName, m_Agent->getLocalConnectionInfo()};
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

AgentConnection const* AgentConnectionManager::recvConnectionAndRequestInfo(
    batch_manager::RequestInfo& requestInfo, std::atomic<bool> const& terminateFlag)
{
    auto const startTime = std::chrono::steady_clock::now();
    auto nextLogTime = startTime + std::chrono::seconds(30);
    while (!terminateFlag.load())
    {
        if (!mIsRunning)
        {
            return nullptr;
        }
        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto const now = std::chrono::steady_clock::now();
        if (now >= nextLogTime)
        {
            size_t pendingNotificationCount = 0;
            for (auto const& [agent, notifications] : mUnhandledNotifications)
            {
                pendingNotificationCount += notifications.size();
            }
            auto const elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            auto const pendingSummary = pendingNotificationsSummary(mUnhandledNotifications);
            TLLM_LOG_INFO(
                "[disagg-debug] C++ recvConnectionAndRequestInfo still waiting: localAgent=%s "
                "pendingAgents=%zu pendingNotifications=%zu elapsedMs=%lld terminate=%d running=%d "
                "pending=[%s]",
                mAgentName.c_str(), mUnhandledNotifications.size(), pendingNotificationCount,
                static_cast<long long>(elapsedMs), static_cast<int>(terminateFlag.load()),
                static_cast<int>(mIsRunning.load()), pendingSummary.c_str());
            nextLogTime = now + std::chrono::seconds(30);
        }
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            auto notifIt = notifs.begin();
            while (notifIt != notifs.end())
            {
                std::stringstream ss(*notifIt);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
                {
                    auto requestAndBufferInfo = std::get<RequestAndBufferInfo>(notificationInfo.mInfo);

                    erase = true;
                    requestInfo = requestAndBufferInfo.mRequestInfo;
                    auto address = requestAndBufferInfo.mAddress;
                    auto bufferDescs = std::move(requestAndBufferInfo.mBufferDescs);
                    auto metadataOpt = requestAndBufferInfo.mMetadata;
                    auto connectionIdx = requestAndBufferInfo.mValidConnectionIdx;
                    auto remoteAgentName = requestAndBufferInfo.mAgentName;
                    TLLM_LOG_INFO(
                        "[disagg-debug] C++ recvConnectionAndRequestInfo matched request-info: localAgent=%s "
                        "notificationAgent=%s remoteAgent=%s requestId=%zu peerSelfIdx=%d connectionIdx=%d "
                        "bufferDescs=%s bufferKinds=%s metadata=%d addressBytes=%zu",
                        mAgentName.c_str(), agent.c_str(), remoteAgentName.c_str(), requestInfo.getRequestId(),
                        requestInfoSelfIdx(requestInfo), connectionIdx, memoryDescsToString(bufferDescs).c_str(),
                        bufferKindsToString(requestAndBufferInfo.mBufferKinds).c_str(),
                        static_cast<int>(metadataOpt.has_value()), address.size());
                    TLLM_LOG_DEBUG(" recv Address:%s", address.c_str());
                    auto connection = connect(remoteAgentName, address, metadataOpt, true);
                    auto bufferKinds = std::move(requestAndBufferInfo.mBufferKinds);

                    std::optional<std::pair<size_t, size_t>> kvOffsetRatio;
                    std::optional<std::pair<size_t, size_t>> rnnOffsetRatio;
                    std::vector<std::pair<size_t, size_t>> offsetRatios;
                    offsetRatios.reserve(bufferDescs.size());

                    for (size_t bi = 0; bi < bufferDescs.size(); bi++)
                    {
                        auto kind = static_cast<batch_manager::BufferKind>(bufferKinds[bi]);
                        switch (kind)
                        {
                        case batch_manager::BufferKind::kKV:
                        case batch_manager::BufferKind::kKV_INDEXER:
                        {
                            if (!kvOffsetRatio)
                            {
                                kvOffsetRatio
                                    = computeSendOffsetRatio(requestInfo.getTransState().getCacheState().value(),
                                        requestInfo.getTransState().getCommState()->getSelfIdx(), mCacheState,
                                        connectionIdx);
                            }
                            offsetRatios.push_back(*kvOffsetRatio);
                            break;
                        }
                        case batch_manager::BufferKind::kRNN:
                        {
                            if (!rnnOffsetRatio)
                            {
                                auto rnnTargetInfo = targetIRanksForRnn(mCacheState,
                                    requestInfo.getTransState().getCacheState().value(),
                                    requestInfo.getTransState().getCommState()->getSelfIdx());
                                size_t rnnOffsetLayer = 0;
                                for (int ri = 0; ri < connectionIdx; ri++)
                                {
                                    rnnOffsetLayer += rnnTargetInfo.getPeerPPDomainLayerNum(ri);
                                }
                                size_t rnnSendLayer = rnnTargetInfo.getPeerPPDomainLayerNum(connectionIdx);
                                rnnOffsetRatio = std::make_pair(rnnOffsetLayer, rnnSendLayer);
                            }
                            offsetRatios.push_back(*rnnOffsetRatio);
                            break;
                        }
                        }
                    }
                    connection->setSenderState(
                        std::move(bufferDescs), connectionIdx, std::move(offsetRatios), std::move(bufferKinds));
                    TLLM_LOG_INFO(
                        "[disagg-debug] C++ recvConnectionAndRequestInfo sender-state ready: localAgent=%s "
                        "remoteAgent=%s requestId=%zu connectionIdx=%d",
                        mAgentName.c_str(), remoteAgentName.c_str(), requestInfo.getRequestId(), connectionIdx);
                    notifIt = notifs.erase(notifIt);
                    if (notifs.empty())
                    {
                        it = mUnhandledNotifications.erase(it);
                    }
                    return connection;
                }

                if (!erase)
                {
                    notifIt++;
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
    auto notifiedSyncMessages = m_Agent->getNotifiedSyncMessages();
    std::lock_guard<std::mutex> lock(mNotificationMutex);

    // Merge new notifications with existing ones
    for (auto const& [agent, notifs] : notifiedSyncMessages)
    {
        if (!notifs.empty())
        {
            auto existingIt = mUnhandledNotifications.find(agent);
            size_t const existingCount = existingIt == mUnhandledNotifications.end() ? 0 : existingIt->second.size();
            std::ostringstream details;
            constexpr size_t kMaxLoggedNotifications = 8;
            for (size_t i = 0; i < notifs.size() && i < kMaxLoggedNotifications; i++)
            {
                if (i > 0)
                {
                    details << ";";
                }
                details << notificationSummary(notifs[i]);
            }
            if (notifs.size() > kMaxLoggedNotifications)
            {
                details << ";...";
            }
            TLLM_LOG_INFO(
                "[disagg-debug] C++ updateUnhandledNotifications: localAgent=%s fromAgent=%s "
                "newNotifications=%zu existingBefore=%zu details=[%s]",
                mAgentName.c_str(), agent.c_str(), notifs.size(), existingCount, details.str().c_str());
        }
        auto& existingNotifications = mUnhandledNotifications[agent];
        existingNotifications.insert(existingNotifications.end(), std::make_move_iterator(notifs.begin()),
            std::make_move_iterator(notifs.end()));
    }
}

[[nodiscard]] std::vector<Connection const*> AgentConnectionManager::getConnections(CommState const& state)
{
    TLLM_CHECK(state.isAgentState());
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

std::vector<batch_manager::BaseTransBufferManager*> const& AgentConnectionManager::getCacheTransBufferManagers() const
{
    return mCacheTransBufferManagers;
}

std::vector<uint8_t> const& AgentConnectionManager::getBufferKinds() const
{
    return mBufferKinds;
}

AgentConnection* AgentConnectionManager::connect(std::string const& remoteAgentName, std::string const& connectionInfo,
    std::optional<std::string> metadata, bool isSender)
{

    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnectionManager connect begin: localAgent=%s remoteAgent=%s "
        "metadata=%d isSender=%d connectionInfoBytes=%zu",
        mAgentName.c_str(), remoteAgentName.c_str(), static_cast<int>(metadata.has_value()), static_cast<int>(isSender),
        connectionInfo.size());
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
            TLLM_LOG_INFO(
                "[disagg-debug] C++ AgentConnectionManager connect loaded existing remote agent: "
                "localAgent=%s remoteAgent=%s",
                mAgentName.c_str(), remoteAgentName.c_str());
        }
        TLLM_LOG_INFO(
            "[disagg-debug] C++ AgentConnectionManager connect reused connection: localAgent=%s "
            "remoteAgent=%s hasLoadRemoteAgent=%d",
            mAgentName.c_str(), remoteAgentName.c_str(), static_cast<int>(it->second->hasLoadRemoteAgent()));
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
            TLLM_CHECK_WITH_INFO(!isSender, "Sender shouldn't call loadRemoteAgent");
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "mAgentName: %s connect to %s with loadRemoteAgent",
                mAgentName.c_str(), remoteAgentName.c_str());
            m_Agent->loadRemoteAgent(remoteAgentName, connectionInfo);
        }
    }
    else
    {
        hasLoadRemoteAgent = true;
    }

    auto connection = std::make_shared<AgentConnection>(mAgentName, remoteAgentName, this);
    mConnections[remoteAgentName] = connection;
    connection->setHasLoadRemoteAgent(hasLoadRemoteAgent);
    TLLM_LOG_INFO(
        "[disagg-debug] C++ AgentConnectionManager connect created connection: localAgent=%s remoteAgent=%s "
        "hasLoadRemoteAgent=%d totalConnections=%zu",
        mAgentName.c_str(), remoteAgentName.c_str(), static_cast<int>(hasLoadRemoteAgent), mConnections.size());
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

template <typename NotificationType>
void AgentConnectionManager::waitForNotification(
    std::string const& remoteAgentName, NotificationType& expectedInfo, std::atomic<bool> const& terminateFlag)
{
    auto const startTime = std::chrono::steady_clock::now();
    auto nextLogTime = startTime + std::chrono::seconds(30);
    char const* notificationType = "unknown";
    if constexpr (std::is_same_v<NotificationType, NotificationSyncInfo>)
    {
        notificationType = "NotificationSyncInfo";
    }
    else if constexpr (std::is_same_v<NotificationType, ReadySignalInfo>)
    {
        notificationType = "ReadySignalInfo";
    }

    while (!terminateFlag.load())
    {

        if (!mIsRunning)
        {
            return;
        }
        updateUnhandledNotifications();
        std::scoped_lock lock(mNotificationMutex);
        auto const now = std::chrono::steady_clock::now();
        if (now >= nextLogTime)
        {
            size_t pendingNotificationCount = 0;
            for (auto const& [agent, notifications] : mUnhandledNotifications)
            {
                pendingNotificationCount += notifications.size();
            }
            auto const elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            TLLM_LOG_INFO(
                "[disagg-debug] C++ waitForNotification still waiting: type=%s remoteAgent=%s "
                "expectedAgent=%s tag=%llu pendingAgents=%zu pendingNotifications=%zu elapsedMs=%lld "
                "terminate=%d running=%d localAgent=%s pending=[%s]",
                notificationType, remoteAgentName.c_str(), expectedInfo.mAgentName.c_str(),
                static_cast<unsigned long long>(expectedInfo.mContext.getTag()), mUnhandledNotifications.size(),
                pendingNotificationCount, static_cast<long long>(elapsedMs), static_cast<int>(terminateFlag.load()),
                static_cast<int>(mIsRunning.load()), mAgentName.c_str(),
                pendingNotificationsSummary(mUnhandledNotifications).c_str());
            nextLogTime = now + std::chrono::seconds(30);
        }
        auto it = mUnhandledNotifications.begin();
        while (it != mUnhandledNotifications.end())
        {
            auto& [agent, notifs] = *it;
            if (agent != remoteAgentName)
            {
                it++;
                continue;
            }
            auto notifIt = notifs.begin();
            while (notifIt != notifs.end())
            {
                std::stringstream ss(*notifIt);
                NotificationInfo notificationInfo = NotificationInfo::deserialize(ss);
                bool erase = false;
                if constexpr (std::is_same_v<NotificationType, NotificationSyncInfo>)
                {
                    if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
                    {
                        auto notificationData = std::get<NotificationSyncInfo>(notificationInfo.mInfo);
                        if (notificationData.mContext.getTag() == expectedInfo.mContext.getTag()
                            && notificationData.mAgentName == expectedInfo.mAgentName)
                        {
                            erase = true;
                            TLLM_LOG_INFO(
                                "[disagg-debug] C++ waitForNotification matched: type=%s remoteAgent=%s "
                                "expectedAgent=%s tag=%llu localAgent=%s",
                                notificationType, remoteAgentName.c_str(), expectedInfo.mAgentName.c_str(),
                                static_cast<unsigned long long>(expectedInfo.mContext.getTag()), mAgentName.c_str());
                            notifIt = notifs.erase(notifIt);
                            if (notifs.empty())
                            {
                                it = mUnhandledNotifications.erase(it);
                            }
                            return;
                        }
                    }
                }
                else if constexpr (std::is_same_v<NotificationType, ReadySignalInfo>)
                {
                    if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
                    {
                        auto readySignalData = std::get<ReadySignalInfo>(notificationInfo.mInfo);
                        if (readySignalData.mContext.getTag() == expectedInfo.mContext.getTag()
                            && readySignalData.mAgentName == expectedInfo.mAgentName)
                        {
                            expectedInfo.mIsReady = readySignalData.mIsReady;

                            erase = true;
                            TLLM_LOG_INFO(
                                "[disagg-debug] C++ waitForNotification matched: type=%s remoteAgent=%s "
                                "expectedAgent=%s tag=%llu isReady=%d localAgent=%s",
                                notificationType, remoteAgentName.c_str(), expectedInfo.mAgentName.c_str(),
                                static_cast<unsigned long long>(expectedInfo.mContext.getTag()),
                                static_cast<int>(expectedInfo.mIsReady), mAgentName.c_str());
                            notifIt = notifs.erase(notifIt);
                            if (notifs.empty())
                            {
                                it = mUnhandledNotifications.erase(it);
                            }
                            return;
                        }
                    }
                }

                if (!erase)
                {
                    notifIt++;
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

// Explicit template instantiations
template void AgentConnectionManager::waitForNotification<NotificationSyncInfo>(
    std::string const& remoteAgentName, NotificationSyncInfo& expectedInfo, std::atomic<bool> const& terminateFlag);
template void AgentConnectionManager::waitForNotification<ReadySignalInfo>(
    std::string const& remoteAgentName, ReadySignalInfo& expectedInfo, std::atomic<bool> const& terminateFlag);

void AgentConnectionManager::waitForSyncInfo(
    std::string const& remoteAgentName, NotificationSyncInfo& syncInfo, std::atomic<bool> const& terminateFlag)
{
    waitForNotification(remoteAgentName, syncInfo, terminateFlag);
}

void AgentConnectionManager::waitForReadySignal(
    std::string const& remoteAgentName, ReadySignalInfo& readySignalInfo, std::atomic<bool> const& terminateFlag)
{
    waitForNotification(remoteAgentName, readySignalInfo, terminateFlag);
}

std::string const& AgentConnectionManager::getAgentName() const
{
    return mAgentName;
}

AgentConnectionManager::~AgentConnectionManager()
{
    mIsRunning = false;
    m_Agent->deregisterMemory(mRegMemDescs);
}

bool AgentConnectionManager::isRunning() const
{
    return mIsRunning;
}

} // namespace tensorrt_llm::executor::kv_cache
