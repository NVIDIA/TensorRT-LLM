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

#pragma once

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include <map>

namespace tensorrt_llm::executor::kv_cache
{
struct RequestAndBufferInfo
{
    std::string mAgentName;
    std::string mAddress;
    batch_manager::RequestInfo mRequestInfo;
    MemoryDesc mBufferDesc;
    std::optional<std::string> mMetadata;
    int mValidConnectionIdx;

    static void serialize(RequestAndBufferInfo const& requestAndBufferInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(requestAndBufferInfo.mAgentName, os);
        su::serialize(requestAndBufferInfo.mAddress, os);
        batch_manager::RequestInfo::serialize(requestAndBufferInfo.mRequestInfo, os);
        MemoryDesc::serialize(requestAndBufferInfo.mBufferDesc, os);
        su::serialize(requestAndBufferInfo.mMetadata, os);
        su::serialize(requestAndBufferInfo.mValidConnectionIdx, os);
    }

    static RequestAndBufferInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentName = su::deserialize<decltype(mAgentName)>(is);
        auto address = su::deserialize<decltype(mAddress)>(is);
        auto requestInfo = batch_manager::RequestInfo::deserialize(is);
        auto bufferDesc = MemoryDesc::deserialize(is);
        auto metadata = su::deserialize<decltype(mMetadata)>(is);
        auto validConnectionIdx = su::deserialize<decltype(mValidConnectionIdx)>(is);
        return RequestAndBufferInfo{agentName, address, requestInfo, bufferDesc, metadata, validConnectionIdx};
    }

    static size_t serializedSize(RequestAndBufferInfo const& requestAndBufferInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(requestAndBufferInfo.mAgentName) + su::serializedSize(requestAndBufferInfo.mAddress)
            + batch_manager::RequestInfo::serializedSize(requestAndBufferInfo.mRequestInfo)
            + MemoryDesc::serializedSize(requestAndBufferInfo.mBufferDesc)
            + su::serializedSize(requestAndBufferInfo.mMetadata)
            + su::serializedSize(requestAndBufferInfo.mValidConnectionIdx);
    }
};

struct ReadySignalInfo
{
    std::string mAgentName;
    DataContext mContext;
    bool mIsReady;

    static void serialize(ReadySignalInfo const& readySignalInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(readySignalInfo.mAgentName, os);
        su::serialize(readySignalInfo.mContext.getTag(), os);
        su::serialize(readySignalInfo.mIsReady, os);
    }

    static ReadySignalInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentName = su::deserialize<decltype(mAgentName)>(is);
        auto contextTag = su::deserialize<decltype(mContext.getTag())>(is);
        DataContext context{contextTag};
        auto isReady = su::deserialize<decltype(mIsReady)>(is);
        return ReadySignalInfo{agentName, context, isReady};
    }

    static size_t serializedSize(ReadySignalInfo const& readySignalInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(readySignalInfo.mAgentName) + su::serializedSize(readySignalInfo.mContext.getTag())
            + su::serializedSize(readySignalInfo.mIsReady);
    }
};

struct NotificationSyncInfo
{

    std::string mAgentName;
    DataContext mContext;

    static void serialize(NotificationSyncInfo const& notificationSyncInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(notificationSyncInfo.mAgentName, os);
        su::serialize(notificationSyncInfo.mContext.getTag(), os);
    }

    static NotificationSyncInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentName = su::deserialize<decltype(mAgentName)>(is);
        auto contextTag = su::deserialize<decltype(mContext.getTag())>(is);
        DataContext context{contextTag};
        return NotificationSyncInfo{agentName, context};
    }

    static size_t serializedSize(NotificationSyncInfo const& notificationSyncInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(notificationSyncInfo.mAgentName)
            + su::serializedSize(notificationSyncInfo.mContext.getTag());
    }
};

struct NotificationInfo
{

    std::variant<RequestAndBufferInfo, NotificationSyncInfo, ReadySignalInfo> mInfo;

    static void serialize(NotificationInfo const& notificationInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(notificationInfo.mInfo.index(), os);
        if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
        {
            RequestAndBufferInfo::serialize(std::get<RequestAndBufferInfo>(notificationInfo.mInfo), os);
        }
        else if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
        {
            NotificationSyncInfo::serialize(std::get<NotificationSyncInfo>(notificationInfo.mInfo), os);
        }
        else if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
        {
            ReadySignalInfo::serialize(std::get<ReadySignalInfo>(notificationInfo.mInfo), os);
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
    }

    static NotificationInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto variantIdx = su::deserialize<std::size_t>(is);
        constexpr std::size_t requestAndBufferInfoIdx{0};
        constexpr std::size_t notificationSyncInfoIdx{1};
        constexpr std::size_t readySignalInfoIdx{2};
        if (variantIdx == requestAndBufferInfoIdx)
        {
            return NotificationInfo{RequestAndBufferInfo::deserialize(is)};
        }
        else if (variantIdx == notificationSyncInfoIdx)
        {
            return NotificationInfo{NotificationSyncInfo::deserialize(is)};
        }
        else if (variantIdx == readySignalInfoIdx)
        {
            return NotificationInfo{ReadySignalInfo::deserialize(is)};
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
    }

    static size_t serializedSize(NotificationInfo const& notificationInfo)
    {
        namespace su = executor::serialize_utils;
        size_t totalSize = 0;
        totalSize += su::serializedSize(notificationInfo.mInfo.index());
        if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
        {
            totalSize += RequestAndBufferInfo::serializedSize(std::get<RequestAndBufferInfo>(notificationInfo.mInfo));
        }
        else if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
        {
            totalSize += NotificationSyncInfo::serializedSize(std::get<NotificationSyncInfo>(notificationInfo.mInfo));
        }
        else if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
        {
            totalSize += ReadySignalInfo::serializedSize(std::get<ReadySignalInfo>(notificationInfo.mInfo));
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
        return totalSize;
    }
};
class AgentConnectionManager;

class AgentConnection : public Connection
{
public:
    AgentConnection(
        std::string mAgentName, std::string mRemoteAgentName, AgentConnectionManager* mAgentConnectionManager);
    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;
    void sendRequestAndBufferInfo(
        batch_manager::RequestInfo& requestInfo, std::optional<size_t> cacheBufferId, int validConnectionIdx);
    void setSenderState(
        MemoryDesc mCacheReceiverBufferDesc, int valideSegmentIdx, std::pair<size_t, size_t> offsetRatio);
    [[nodiscard]] std::optional<size_t> getCacheBufferId() const;
    void setHasLoadRemoteAgent(bool hasLoadRemoteAgent);
    [[nodiscard]] bool hasLoadRemoteAgent() const;
    void sendReadySignal(DataContext const& ctx, bool isReady) const;
    bool recvReadySignal(DataContext const& ctx) const;

private:
    std::string mAgentName;
    std::string mRemoteAgentName;

    struct SenderState
    {
        MemoryDesc mCacheReceiverBufferDesc{nullptr, 0, 0};
        int validSegmentIdx{0};
        std::pair<size_t, size_t> mOffsetRatio;
        SenderState() = default;
    };

    AgentConnectionManager* mAgentConnectionManager;

    batch_manager::kv_cache_manager::CacheTransBufferManager* mCacheTransBufferManager;
    std::optional<size_t> mCacheBufferId;
    SenderState mSenderState;
    bool mNeedSendMetadata{true};
    bool mHasLoadRemoteAgent{false};
};

class AgentConnectionManager : public ConnectionManager
{
public:
    AgentConnectionManager(
        batch_manager::kv_cache_manager::CacheTransBufferManager* cacheTransBufferManager, CacheState cacheState);
    ~AgentConnectionManager();
    AgentConnection* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    [[nodiscard]] std::vector<Connection const*> getConnections(CommState const& state) override;
    [[nodiscard]] CommState const& getCommState() const override;
    AgentConnection const* recvConnectionAndRequestInfo(batch_manager::RequestInfo& requestInfo);
    [[nodiscard]] batch_manager::kv_cache_manager::CacheTransBufferManager* getCacheTransBufferManager();
    void updateUnhandledNotifications();
    [[nodiscard]] BaseTransferAgent* getAgent() const;
    AgentConnection* connect(std::string const& remoteAgentName, std::string const& address,
        std::optional<std::string> metadata = std::nullopt, bool isSender = false);
    int getDeviceId() const;
    [[nodiscard]] std::string const& getAgentName() const;

    template <typename NotificationType>
    void waitForNotification(std::string const& remoteAgentName, NotificationType& expectedInfo);
    void waitForSyncInfo(std::string const& remoteAgentName, NotificationSyncInfo& syncInfo);
    void waitForReadySignal(std::string const& remoteAgentName, ReadySignalInfo& readySignalInfo);

private:
    std::map<std::string, std::shared_ptr<AgentConnection>> mConnections;
    std::mutex mConnectionsMutex;
    CommState mCommState;
    CacheState mCacheState;
    batch_manager::kv_cache_manager::CacheTransBufferManager* mCacheTransBufferManager;
    std::mutex mNotificationMutex;
    std::unordered_map<std::string, std::list<std::string>> mUnhandledNotifications;
    std::unique_ptr<BaseTransferAgent> m_Agent;
    int mDeviceId;
    std::string mAgentName;
    MemoryDescs mRegMemDescs;
};

} // namespace tensorrt_llm::executor::kv_cache
