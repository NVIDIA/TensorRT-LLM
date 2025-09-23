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
#include <vector>

std::vector<std::string> findLocalIpAddresses()
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

namespace tensorrt_llm::executor::kv_cache
{

MooncakeTransferAgent::MooncakeTransferAgent(BaseAgentConfig const& config)
{
    local_agent_name_ = config.mName;
    auto ips = findLocalIpAddresses();
    segment_name_ = "127.0.0.1";
    if (!ips.empty())
        segment_name_ = ips[0];
    // if (getenv("NIXL_MOONCAKE_IP_ADDR"))
    //    segment_name_ = std::string(getenv("NIXL_MOONCAKE_IP_ADDR"));
    engine_ = createTransferEngine("P2PHANDSHAKE", segment_name_.c_str(), "", 0, true);
}

void MooncakeTransferAgent::registerMemory(RegisterDescs const& descs) {}

void MooncakeTransferAgent::deregisterMemory(RegisterDescs const& descs) {}

void MooncakeTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc) {}

AgentDesc MooncakeTransferAgent::getLocalAgentDesc() {}

void MooncakeTransferAgent::invalidateRemoteAgent(std::string const& name) {}

[[nodiscard]] std::unique_ptr<TransferStatus> MooncakeTransferAgent::submitTransferRequests(
    TransferRequest const& request)
{
}

void MooncakeTransferAgent::notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) {}

[[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> MooncakeTransferAgent::getNotifiedSyncMessages()
{
}

ConnectionInfoType MooncakeTransferAgent::getConnectionInfo()
{
    return segment_name_;
}

void MooncakeTransferAgent::connectRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) {}

bool MooncakeTransferAgent::checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs)
{
    return true;
}

MooncakeTransferAgent::~MooncakeTransferAgent()
{
    destroyTransferEngine(engine_);
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
