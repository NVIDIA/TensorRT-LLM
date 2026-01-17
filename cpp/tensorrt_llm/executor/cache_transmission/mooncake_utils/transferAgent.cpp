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

#include "tensorrt_llm/executor/cache_transmission/mooncake_utils/transferAgent.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/ipUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

MooncakeTransferStatus::MooncakeTransferStatus(transfer_engine_t engine, uint64_t batchId, size_t requestCount)
    : mEngine{engine}
    , mBatchId{batchId}
    , mRequestCount{requestCount}
{
    TLLM_CHECK(mEngine);
}

TransferState MooncakeTransferStatus::wait(int64_t timeout_ms) const
{
    auto startTime = std::chrono::steady_clock::now();

    while (true)
    {
        if (mBatchFreed)
        {
            return TransferState::kSUCCESS;
        }

        bool has_failed = false;
        bool all_completed = true;

        for (size_t index = 0; index < mRequestCount; ++index)
        {
            transfer_status_t status;
            int rc = getTransferStatus(mEngine, mBatchId, index, &status);
            if (rc || status.status == STATUS_FAILED)
            {
                has_failed = true;
                if (rc)
                {
                    TLLM_LOG_ERROR(
                        "Failed to get transfer status for batch %lu, task %zu: error code %d", mBatchId, index, rc);
                }
                else
                {
                    TLLM_LOG_ERROR(
                        "Transfer failed for batch %lu, task %zu: status %d", mBatchId, index, status.status);
                }
            }
            else if (status.status != STATUS_COMPLETED)
            {
                all_completed = false;
            }
        }

        // If any request failed, return failure
        if (has_failed)
        {
            return TransferState::kFAILURE;
        }

        // If all requests completed successfully
        if (all_completed)
        {
            freeBatchID(mEngine, mBatchId);
            mBatchFreed = true;
            TLLM_LOG_DEBUG("Batch ID %lu freed in wait()", mBatchId);
            syncSegmentCache(mEngine);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return TransferState::kSUCCESS;
        }

        // If timeout_ms < 0, wait indefinitely
        if (timeout_ms < 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Check if timeout has elapsed
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        if (elapsed >= timeout_ms)
        {
            return TransferState::kIN_PROGRESS;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

[[nodiscard]] bool MooncakeTransferStatus::isCompleted() const
{
    if (mBatchFreed)
    {
        return true;
    }

    bool has_failed = false;
    for (size_t index = 0; index < mRequestCount; ++index)
    {
        transfer_status_t status;
        int rc = getTransferStatus(mEngine, mBatchId, index, &status);
        if (rc || status.status == STATUS_FAILED)
        {
            has_failed = true;
            if (rc)
            {
                TLLM_LOG_ERROR(
                    "Failed to get transfer status for batch %lu, task %zu: error code %d", mBatchId, index, rc);
            }
            else
            {
                TLLM_LOG_ERROR("Transfer failed for batch %lu, task %zu: status %d", mBatchId, index, status.status);
            }
        }
        else if (status.status == STATUS_PENDING || status.status == STATUS_WAITING)
        {
            TLLM_LOG_DEBUG("Transfer is pending for batch %lu, task %zu", mBatchId, index);
            return false;
        }
    }
    if (!has_failed)
    {
        // Each batchId has the batch size, and cannot process more requests
        // than the batch size. So, free the batch id here to workaround the issue
        // where the same batchId could be used to post multiple transfer.
        freeBatchID(mEngine, mBatchId);
        mBatchFreed = true;
        TLLM_LOG_DEBUG("Batch ID %lu freed, future calls will return true directly", mBatchId);
    }
    // Currently, we cannot distinguish between failed and completed from return value.
    TLLM_LOG_DEBUG("Transfer is completed for batch %lu", mBatchId);
    return true;
}

std::string const MooncakeBase64Helper::STANDARD_CHARS
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "0123456789+/";

std::string MooncakeBase64Helper::encode(std::vector<uint8_t> const& data)
{
    return encodeInternal(data, STANDARD_CHARS);
}

std::string MooncakeBase64Helper::encode(std::string const& data)
{
    std::vector<uint8_t> vec(data.begin(), data.end());
    return encode(vec);
}

std::vector<uint8_t> MooncakeBase64Helper::decode(std::string const& encoded)
{
    return decodeInternal(encoded, STANDARD_CHARS);
}

std::string MooncakeBase64Helper::decodeToString(std::string const& encoded)
{
    auto vec = decode(encoded);
    return std::string(vec.begin(), vec.end());
}

std::string MooncakeBase64Helper::encodeInternal(std::vector<uint8_t> const& data, std::string const& chars)
{
    std::string encoded;
    size_t i = 0;
    size_t j = 0;
    std::array<uint8_t, 3> charArray3{};
    std::array<uint8_t, 4> charArray4{};
    size_t dataLen = data.size();
    uint8_t const* bytes = data.data();

    while (dataLen--)
    {
        charArray3[i++] = *(bytes++);
        if (i == 3)
        {
            charArray4[0] = (charArray3[0] & 0xfc) >> 2;
            charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
            charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
            charArray4[3] = charArray3[2] & 0x3f;

            for (i = 0; i < 4; i++)
            {
                encoded += chars[charArray4[i]];
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 3; j++)
        {
            charArray3[j] = '\0';
        }

        charArray4[0] = (charArray3[0] & 0xfc) >> 2;
        charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
        charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
        charArray4[3] = charArray3[2] & 0x3f;

        for (j = 0; j < i + 1; j++)
        {
            encoded += chars[charArray4[j]];
        }

        while (i++ < 3)
        {
            encoded += '=';
        }
    }

    return encoded;
}

std::vector<uint8_t> MooncakeBase64Helper::decodeInternal(std::string const& encoded, std::string const& chars)
{
    size_t encodedLen = encoded.size();
    size_t i = 0;
    size_t j = 0;
    size_t in_ = 0;
    std::array<uint8_t, 3> charArray3{};
    std::array<uint8_t, 4> charArray4{};
    std::vector<uint8_t> decoded;

    std::string cleanEncoded;
    for (char c : encoded)
    {
        if (!isWhitespace(c))
        {
            cleanEncoded += c;
        }
    }

    encodedLen = cleanEncoded.size();

    while (encodedLen-- && cleanEncoded[in_] != '=' && isBase64(cleanEncoded[in_], chars))
    {
        charArray4[i++] = cleanEncoded[in_];
        in_++;
        if (i == 4)
        {
            for (i = 0; i < 4; i++)
            {
                charArray4[i] = chars.find(charArray4[i]);
            }

            charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
            charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
            charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

            for (i = 0; i < 3; i++)
            {
                decoded.push_back(charArray3[i]);
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 4; j++)
        {
            charArray4[j] = 0;
        }

        for (j = 0; j < 4; j++)
        {
            charArray4[j] = chars.find(charArray4[j]);
        }

        charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
        charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
        charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

        for (j = 0; j < i - 1; j++)
        {
            decoded.push_back(charArray3[j]);
        }
    }

    return decoded;
}

bool MooncakeBase64Helper::isBase64(uint8_t c, std::string const& chars)
{
    return (isalnum(c) || (c == chars[62]) || (c == chars[63]));
}

bool MooncakeBase64Helper::isWhitespace(uint8_t c)
{
    return (c == ' ' || c == '\n' || c == '\r' || c == '\t');
}

MooncakeTransferAgent::MooncakeTransferAgent(BaseAgentConfig const& config)
{
    mLocalAgentName = config.mName;
    std::string segmentName = "127.0.0.1";

    if (getenv("TLLM_MOONCAKE_IP_ADDR"))
    {
        segmentName = std::string(getenv("TLLM_MOONCAKE_IP_ADDR"));
    }
    else
    {
        auto ip = common::getLocalIp(common::getEnvMooncakeInterface(), mpi::MpiComm::session().getRank());
        if (!ip.empty())
            segmentName = ip;
    }

    mEngine = createTransferEngine("P2PHANDSHAKE", segmentName.c_str(), "", 0, true);
}

void MooncakeTransferAgent::registerMemory(RegisterDescs const& descs)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::registerMemory");

    std::lock_guard<std::mutex> lock(mMutex);
    for (auto const& desc : descs.getDescs())
    {
        auto it = mMemRegInfo.find(desc.getAddr());
        if (it != mMemRegInfo.end())
        {
            it->second->addRef();
            continue;
        }

        int err = registerLocalMemory(mEngine, reinterpret_cast<void*>(desc.getAddr()), desc.getLen(), "*", 1);

        TLLM_CHECK_WITH_INFO(err == 0, "registerLocalMemory failed, addr: %p, len: %lu",
            reinterpret_cast<void*>(desc.getAddr()), desc.getLen());

        auto mooncakeDesc = std::make_shared<MooncakeMemoryDesc>(desc);
        mMemRegInfo[desc.getAddr()] = std::move(mooncakeDesc);
    }
}

void MooncakeTransferAgent::deregisterMemory(RegisterDescs const& descs)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::deregisterMemory");

    std::lock_guard<std::mutex> lock(mMutex);
    for (auto const& desc : descs.getDescs())
    {
        auto it = mMemRegInfo.find(desc.getAddr());
        if (it != mMemRegInfo.end())
        {
            auto const& mooncakeDesc = it->second;
            mooncakeDesc->releaseRef();
            if (mooncakeDesc->getRefCount())
                continue;

            int err = unregisterLocalMemory(mEngine, reinterpret_cast<void*>(desc.getAddr()));

            TLLM_CHECK_WITH_INFO(
                err == 0, "unregisterLocalMemory failed, addr: %p", reinterpret_cast<void*>(desc.getAddr()));

            mMemRegInfo.erase(desc.getAddr());
        }
    }
}

void MooncakeTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::loadRemoteAgent");

    // Do the same thing as loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo)
    loadRemoteAgent(name, std::move(agentDesc.getBackendAgentDesc()));
}

void MooncakeTransferAgent::loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo)
{
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "MooncakeTransferAgent::loadRemoteAgent loadRemoteAgent to %s remoteagent name: %s", connectionInfo.c_str(),
        name.c_str());

    std::lock_guard<std::mutex> lock(mMutex);
    auto segmentId = openSegment(mEngine, connectionInfo.c_str());

    TLLM_CHECK_WITH_INFO(
        segmentId >= 0, "loadRemoteAgent openSegment failed, connectionInfo: %s", connectionInfo.c_str());

    mConnectedAgents[name].segmentId = segmentId;
}

void MooncakeTransferAgent::invalidateRemoteAgent(std::string const& name)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::invalidateRemoteAgent");
}

AgentDesc MooncakeTransferAgent::getLocalAgentDesc()
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::getLocalAgentDesc");

    // Using connection info as agent desc
    static size_t const kBufLen = 64;
    char connectionInfo[kBufLen];

    int ret = getLocalIpAndPort(mEngine, connectionInfo, kBufLen);

    TLLM_CHECK_WITH_INFO(ret == 0, "MooncakeTransferAgent::getLocalAgentDesc::getLocalIpAndPort failed");

    return AgentDesc{std::string(connectionInfo)};
}

ConnectionInfoType MooncakeTransferAgent::getLocalConnectionInfo()
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::getLocalConnectionInfo");

    static size_t const kBufLen = 64;
    char connectionInfo[kBufLen];

    int ret = getLocalIpAndPort(mEngine, connectionInfo, kBufLen);

    TLLM_CHECK_WITH_INFO(ret == 0, "MooncakeTransferAgent::getLocalAgentDesc::getLocalConnectionInfo failed");

    return std::string(connectionInfo);
}

[[nodiscard]] std::unique_ptr<TransferStatus> MooncakeTransferAgent::submitTransferRequests(
    TransferRequest const& request)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::submitTransferRequests");

    bool hasNotif = false;
    std::string syncMessage;

    if (request.getSyncMessage().has_value())
    {
        hasNotif = true;
        syncMessage = request.getSyncMessage().value();
    }

    static size_t const kMaxRequestCount = 1024;
    uint64_t batchId = allocateBatchID(mEngine, kMaxRequestCount);

    TLLM_CHECK_WITH_INFO(batchId != INVALID_BATCH, "allocateBatchID failed");

    int segmentId;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        std::string remoteName = request.getRemoteName();

        auto it = mConnectedAgents.find(remoteName);
        if (it == mConnectedAgents.end())
        {
            std::string error = "Remote agent " + remoteName + "not found";
            TLLM_THROW(error);
        }

        auto const& agentInfo = it->second;
        segmentId = agentInfo.segmentId;
    }

    auto localDescs = request.getSrcDescs().getDescs();
    auto remoteDescs = request.getDstDescs().getDescs();

    TLLM_CHECK_WITH_INFO(localDescs.size() == remoteDescs.size(), "Number of local and remote memory must match");

    size_t requestCount = localDescs.size();
    std::vector<transfer_request_t> transferRequests(requestCount);

    for (size_t index = 0; index < requestCount; ++index)
    {
        TLLM_CHECK_WITH_INFO(
            localDescs[index].getLen() == remoteDescs[index].getLen(), "Length of local and remote memory must match");

        transferRequests[index].opcode = (request.getOp() == TransferOp::kREAD) ? OPCODE_READ : OPCODE_WRITE;
        transferRequests[index].source = reinterpret_cast<void*>(localDescs[index].getAddr());
        transferRequests[index].target_offset = remoteDescs[index].getAddr();
        transferRequests[index].length = localDescs[index].getLen();
        transferRequests[index].target_id = segmentId;
    }

    int rc = 0;
    if (hasNotif)
    {
        notify_msg_t notifyMsg;
        notifyMsg.name = const_cast<char*>(mLocalAgentName.c_str());
        notifyMsg.msg = const_cast<char*>(syncMessage.c_str());
        rc = submitTransferWithNotify(mEngine, batchId, transferRequests.data(), requestCount, notifyMsg);
    }
    else
    {
        rc = submitTransfer(mEngine, batchId, transferRequests.data(), requestCount);
    }

    TLLM_CHECK_WITH_INFO(rc == 0, "submitTransfer failed with status: %d", rc);

    return std::make_unique<MooncakeTransferStatus>(mEngine, batchId, requestCount);
}

void MooncakeTransferAgent::notifySyncMessage(std::string const& name, SyncMessage const& syncMessage)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::notifySyncMessage");
    int segmentId;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mConnectedAgents.find(name);

        if (it == mConnectedAgents.end())
        {
            TLLM_LOG_WARNING("Remote agent %s not found", name.c_str());
            return;
        }

        auto const& agentInfo = it->second;
        segmentId = agentInfo.segmentId;
    }

    notify_msg_t notifyMsg;
    notifyMsg.name = const_cast<char*>(mLocalAgentName.c_str());
    std::string encoded = MooncakeBase64Helper::encode(syncMessage);
    notifyMsg.msg = const_cast<char*>(encoded.c_str());

    TLLM_LOG_DEBUG("MooncakeTransferAgent::notifySyncMessage notifyMsg.name: %s, notifyMsg.msg: %s", notifyMsg.name,
        notifyMsg.msg);

    int ret = genNotifyInEngine(mEngine, segmentId, notifyMsg);

    TLLM_CHECK_WITH_INFO(ret == 0, "genNotifyInEngine failed with status: %d", ret);
}

[[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> MooncakeTransferAgent::getNotifiedSyncMessages()
{
    std::unordered_map<std::string, std::vector<SyncMessage>> notifs;
    int size = 0;

    notify_msg_t* notifyMsgs = getNotifsFromEngine(mEngine, &size);

    TLLM_CHECK_WITH_INFO(size >= 0, "getNotifsFromEngine returned negative size: %d", size);

    for (int i = 0; i < size; i++)
    {
        if (notifyMsgs[i].msg == nullptr)
        {
            TLLM_LOG_WARNING("Message pointer is null for: %s", notifyMsgs[i].name);
            continue;
        }

        std::string decoded = MooncakeBase64Helper::decodeToString(notifyMsgs[i].msg);
        notifs[notifyMsgs[i].name].emplace_back(std::move(decoded));

        TLLM_LOG_DEBUG("MooncakeTransferAgent::getNotifiedSyncMessages getNotifsFromEngine: %s, %s", notifyMsgs[i].name,
            notifyMsgs[i].msg);
    }

    freeNotifsMsgBuf(notifyMsgs, size);
    return notifs;
}

bool MooncakeTransferAgent::checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::checkRemoteDescs");
    return true;
}

MooncakeTransferAgent::~MooncakeTransferAgent()
{
    destroyTransferEngine(mEngine);
    TLLM_LOG_DEBUG("MooncakeTransferAgent::~MooncakeTransferAgent");
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    std::unique_ptr<BaseTransferAgent> createMooncakeTransferAgent(BaseAgentConfig const* config)
    {
        TLLM_CHECK(config);
        return std::make_unique<MooncakeTransferAgent>(*config);
    }
}

} // namespace tensorrt_llm::executor::kv_cache
