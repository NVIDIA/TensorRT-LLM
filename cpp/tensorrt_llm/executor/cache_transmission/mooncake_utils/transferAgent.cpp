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
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <algorithm>
#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/file.h>
#include <sys/stat.h>
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

void MooncakeTransferStatus::wait() const
{
    while (!isCompleted())
        ;
}

[[nodiscard]] bool MooncakeTransferStatus::isCompleted() const
{
    bool has_failed = false;
    for (size_t index = 0; index < mRequestCount; ++index)
    {
        transfer_status_t status;
        int rc = getTransferStatus(mEngine, mBatchId, index, &status);
        if (rc || status.status == STATUS_FAILED)
            has_failed = true;
        else if (status.status == STATUS_PENDING || status.status == STATUS_WAITING)
            return false;
    }
    if (!has_failed)
    {
        // Each batchId has the batch size, and cannot process more requests
        // than the batch size. So, free the batch id here to workaround the issue
        // where the same batchId could be used to post multiple transfer.
        freeBatchID(mEngine, mBatchId);
        // mBatchId = INVALID_BATCH;
    }
    // Currently, we cannot distinguish between failed and completed
    return true;
}

const std::string MooncakeBase64Helper::STANDARD_CHARS
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
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    size_t data_len = data.size();
    uint8_t const* bytes = data.data();

    while (data_len--)
    {
        char_array_3[i++] = *(bytes++);
        if (i == 3)
        {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++)
            {
                encoded += chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 3; j++)
        {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; j < i + 1; j++)
        {
            encoded += chars[char_array_4[j]];
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
    size_t encoded_len = encoded.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<uint8_t> decoded;

    std::string clean_encoded;
    for (char c : encoded)
    {
        if (!is_whitespace(c))
        {
            clean_encoded += c;
        }
    }

    encoded_len = clean_encoded.size();

    while (encoded_len-- && clean_encoded[in_] != '=' && is_base64(clean_encoded[in_], chars))
    {
        char_array_4[i++] = clean_encoded[in_];
        in_++;
        if (i == 4)
        {
            for (i = 0; i < 4; i++)
            {
                char_array_4[i] = chars.find(char_array_4[i]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++)
            {
                decoded.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i > 0)
    {
        for (j = i; j < 4; j++)
        {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++)
        {
            char_array_4[j] = chars.find(char_array_4[j]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++)
        {
            decoded.push_back(char_array_3[j]);
        }
    }

    return decoded;
}

bool MooncakeBase64Helper::is_base64(unsigned char c, std::string const& chars)
{
    return (isalnum(c) || (c == chars[62]) || (c == chars[63]));
}

bool MooncakeBase64Helper::is_whitespace(unsigned char c)
{
    return (c == ' ' || c == '\n' || c == '\r' || c == '\t');
}

static std::vector<std::string> findLocalIpAddresses()
{
    std::vector<std::string> ips;
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1)
    {
        return ips;
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr == nullptr)
        {
            continue;
        }

        if (ifa->ifa_addr->sa_family == AF_INET)
        {
            if (strcmp(ifa->ifa_name, "lo") == 0)
            {
                continue;
            }

            // Check if interface is UP and RUNNING
            if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING))
            {
                TLLM_LOG_INFO("Skipping interface %s (not UP or not RUNNING)", ifa->ifa_name);
                continue;
            }

            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST)
                == 0)
            {
                ips.push_back(host);
            }
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

MooncakeTransferAgent::MooncakeTransferAgent(BaseAgentConfig const& config)
{
    mLocalAgentName = config.mName;
    std::string segmentName = "127.0.0.1";

    auto ips = findLocalIpAddresses();
    if (!ips.empty())
        segmentName = ips[0];

    if (getenv("TLLM_MOONCAKE_IP_ADDR"))
        segmentName = std::string(getenv("TLLM_MOONCAKE_IP_ADDR"));

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

        int err = registerLocalMemory(mEngine, (void*) desc.getAddr(), desc.getLen(), "*", 1);

        TLLM_CHECK_WITH_INFO(
            err == 0, "registerLocalMemory failed, addr: %p, len: %lu", (void*) desc.getAddr(), desc.getLen());

        auto mooncakeDesc = std::make_shared<MooncakeMemoryDesc>(desc);
        mMemRegInfo[desc.getAddr()] = mooncakeDesc;
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
            auto mooncakeDesc = it->second;
            mooncakeDesc->releaseRef();
            if (mooncakeDesc->getRefCount())
                continue;

            int err = unregisterLocalMemory(mEngine, (void*) desc.getAddr());

            TLLM_CHECK_WITH_INFO(err == 0, "unregisterLocalMemory failed, addr: %p", (void*) desc.getAddr());

            mMemRegInfo.erase(desc.getAddr());
        }
    }
}

void MooncakeTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::loadRemoteAgent");

    // Do the same thing as connectRemoteAgent
    connectRemoteAgent(name, std::move(agentDesc.getBackendAgentDesc()));
}

AgentDesc MooncakeTransferAgent::getLocalAgentDesc()
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::getLocalAgentDesc");

    // Using connection info as agent desc
    const static size_t kBufLen = 64;
    char connectionInfo[kBufLen];

    int ret = getLocalIpAndPort(mEngine, connectionInfo, kBufLen);

    TLLM_CHECK_WITH_INFO(ret == 0, "MooncakeTransferAgent::getLocalAgentDesc::getLocalIpAndPort failed");

    return AgentDesc{std::string(connectionInfo)};
}

void MooncakeTransferAgent::invalidateRemoteAgent(std::string const& name)
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::invalidateRemoteAgent");
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

    const static size_t kMaxRequestCount = 1024;
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

        auto agentInfo = it->second;
        segmentId = agentInfo.segmentId;
    }

    auto localDescs = request.getSrcDescs().getDescs();
    auto remoteDescs = request.getDstDescs().getDescs();

    TLLM_CHECK_WITH_INFO(localDescs.size() == remoteDescs.size(), "Number of local and remote memory must match");

    size_t requestCount = localDescs.size();
    transfer_request_t* transferRequests = new transfer_request_t[requestCount];

    for (size_t index = 0; index < requestCount; ++index)
    {
        TLLM_CHECK_WITH_INFO(
            localDescs[index].getLen() == remoteDescs[index].getLen(), "Length of local and remote memory must match");

        transferRequests[index].opcode = (request.getOp() == TransferOp::kREAD) ? OPCODE_READ : OPCODE_WRITE;
        transferRequests[index].source = (void*) localDescs[index].getAddr();
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
        rc = submitTransferWithNotify(mEngine, batchId, transferRequests, requestCount, notifyMsg);
    }
    else
    {
        rc = submitTransfer(mEngine, batchId, transferRequests, requestCount);
    }

    delete[] transferRequests;

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

        auto agentInfo = it->second;
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
        notifs[notifyMsgs[i].name].push_back(decoded);

        TLLM_LOG_DEBUG("MooncakeTransferAgent::getNotifiedSyncMessages getNotifsFromEngine: %s, %s", notifyMsgs[i].name,
            notifyMsgs[i].msg);
    }

    free(notifyMsgs);
    return notifs;
}

ConnectionInfoType MooncakeTransferAgent::getConnectionInfo()
{
    TLLM_LOG_DEBUG("MooncakeTransferAgent::getConnectionInfo");

    const static size_t kBufLen = 64;
    char connectionInfo[kBufLen];

    int ret = getLocalIpAndPort(mEngine, connectionInfo, kBufLen);

    TLLM_CHECK_WITH_INFO(ret == 0, "MooncakeTransferAgent::getLocalAgentDesc::getConnectionInfo failed");

    return std::string(connectionInfo);
}

void MooncakeTransferAgent::connectRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo)
{
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "MooncakeTransferAgent::connectRemoteAgent connectRemoteAgent to %s remoteagent name: %s",
        connectionInfo.c_str(), name.c_str());

    std::lock_guard<std::mutex> lock(mMutex);
    auto segmentId = openSegment(mEngine, connectionInfo.c_str());

    TLLM_CHECK_WITH_INFO(
        segmentId >= 0, "connectRemoteAgent openSegment failed, connectionInfo: %s", connectionInfo.c_str());

    mConnectedAgents[name].segmentId = segmentId;
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
