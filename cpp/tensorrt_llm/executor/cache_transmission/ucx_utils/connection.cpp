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

#if ENABLE_UCX

#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"

namespace tensorrt_llm::executor::kv_cache
{

UcxConnection::UcxConnection(
    uint64_t connectionId, std::shared_ptr<ucxx::Endpoint> endpoint, std::shared_ptr<UcxConnectionManager> manager)
    : mConnectionId(connectionId)
    , mEndpoint(endpoint)
    , mManager(std::move(manager))
{
    if (mEndpoint)
    {
        initializeEndpointTag();
    }
}

// UcxConnection::UcxConnection(UcxConnection&& other): mConnectionId(std::move(other.mConnectionId)),
// mEndpoint(std::move(other.mEndpoint),mManager(other.mManager){
// }

void UcxConnection::initializeEndpointTag(int maxTryTimes)
{
    // [FIXME] IP exchange seems to be more robust to ensure tag establishment,
    // i.e. different peers can use the same port number to connect with self worker
    // which results in identical tag if only self / peer port is used.

    // knowing that ucxx::Tag is uint64_t
    ucxx::Tag localPort{0};
    ucxx::Tag remotePort{0};
    char lIpStr[INET6_ADDRSTRLEN];
    char rIpStr[INET6_ADDRSTRLEN];
    char portStr[INET6_ADDRSTRLEN];

    ucp_ep_attr_t ep_attr;
    ep_attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR | UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;

    ucs_status_t status = ucp_ep_query(mEndpoint->getHandle(), &ep_attr);
    if (status == UCS_OK)
    {

        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.remote_sockaddr, rIpStr, portStr, INET6_ADDRSTRLEN);
        remotePort = static_cast<ucxx::Tag>(std::stoull(portStr));

        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.local_sockaddr, lIpStr, portStr, INET6_ADDRSTRLEN);
        localPort = static_cast<ucxx::Tag>(std::stoull(portStr));

        // Network port value is defined to fit in 16 bit.
        // sendTag format : [remotePort localPort remotIP]
        mSendTag = static_cast<ucxx::Tag>((localPort << (16 + 32)) | (remotePort << 32) | std::stoull(lIpStr));
        mRecvTag = static_cast<ucxx::Tag>((remotePort << (16 + 32)) | (localPort << 32) | std::stoull(rIpStr));
    }
    else
    {
        // [FIXME] better message
        if (status == UCS_ERR_NOT_CONNECTED && maxTryTimes > 0)
        {
            TLLM_LOG_WARNING("UCX connection has not been established yet. wait 100 ms before retrying. maxTryTimes:%d",
                maxTryTimes);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            initializeEndpointTag(maxTryTimes - 1);
        }
        else
        {
            TLLM_LOG_WARNING("UCX data transceiver is not created by connecting to a socket address.");
        }
    }
}

void UcxConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req = mEndpoint->tagSend(const_cast<void*>(data), size, mSendTag, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
}

void UcxConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req = mEndpoint->tagRecv(data, size, mRecvTag, ucxx::TagMaskFull, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    // throw if there is error
    req->checkError();
}

// UcxServer::UcxServer(uint16_t listenerPort)
// {
//     mContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
//     mWorker = mContext->createWorker();
//     // Ensure the progress thread has CUDA context initialized
//     int device;
//     TLLM_CUDA_CHECK(cudaGetDevice(&device));
//     mWorker->setProgressThreadStartCallback([device](void* arg) { TLLM_CUDA_CHECK(cudaSetDevice(device)); },
//     nullptr); mWorker->startProgressThread(); startListener(listenerPort);
// }

// void UcxServer::listenerCallback(ucp_conn_request_h conn_request, void* arg)
// {
//     auto* self = reinterpret_cast<UcxServer*>(arg);
//     auto endpoint = self->mListener->createEndpointFromConnRequest(conn_request);
//     // TLLM_CHECK(self->mManager.insert("default", std::make_unique<UcxConnection>(std::move(endpoint))));
// }

// void UcxServer::startListener(uint16_t listenerPort)
// {
//     mListener = mWorker->createListener(listenerPort, listenerCallback, this);
// #if __linux__
//     // query network interface

//     struct ifaddrs *ifa, *ifaddr;
//     void* tmpAddrPtr;

//     TLLM_CHECK_WITH_INFO(getifaddrs(&ifaddr) == 0, " UCX startListener getifaddrs call failed\n");
//     TLLM_CHECK_WITH_INFO((ifaddr != NULL), "UCX startListener getifaddrs call failed\n");
//     int idx = 0;
//     for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
//     {
//         // exclude docker interface and loopback
//         if (strcmp(ifa->ifa_name, "docker0") == 0 || strcmp(ifa->ifa_name, "lo") == 0)
//         {
//             continue;
//         }
//         if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET)
//         {
//             tmpAddrPtr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
//             char buffer[INET_ADDRSTRLEN];
//             inet_ntop(AF_INET, tmpAddrPtr, buffer, INET_ADDRSTRLEN);
//             mNetInfoMap[std::string(ifa->ifa_name)].interface = std::string(ifa->ifa_name);
//             mNetInfoMap[std::string(ifa->ifa_name)].ipv4 = std::string(buffer);
//             if (mNetInfoMap[std::string(ifa->ifa_name)].idx == -1)
//             {
//                 mNetInfoMap[std::string(ifa->ifa_name)].idx = idx;
//                 idx++;
//             }
//         }
//         else if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET6)
//         {
//             tmpAddrPtr = &((struct sockaddr_in6*) ifa->ifa_addr)->sin6_addr;
//             char buffer[INET6_ADDRSTRLEN];
//             inet_ntop(AF_INET6, tmpAddrPtr, buffer, INET6_ADDRSTRLEN);
//             mNetInfoMap[std::string(ifa->ifa_name)].interface = ifa->ifa_name;
//             mNetInfoMap[std::string(ifa->ifa_name)].ipv6 = std::string(buffer);
//             if (mNetInfoMap[std::string(ifa->ifa_name)].idx == -1)
//             {
//                 mNetInfoMap[std::string(ifa->ifa_name)].idx = idx;
//                 idx++;
//             }
//         }
//     }
//     std::string selectedIp;
//     std::string userUCXInterface = common::getEnvUCXInterface();
//     if (!userUCXInterface.empty())
//     {
//         if (mNetInfoMap.find(userUCXInterface) != mNetInfoMap.end())
//         {
//             selectedIp = mNetInfoMap[userUCXInterface].ipv4;
//             if (selectedIp.empty())
//             {
//                 selectedIp = mNetInfoMap[userUCXInterface].ipv6;
//             }
//             TLLM_LOG_INFO("UCX listener started on interface:%s address: %s:%u",
//                 mNetInfoMap[userUCXInterface].interface.c_str(), selectedIp.c_str(), mListener->getPort());
//             freeifaddrs(ifaddr);
//             return;
//         }

//         TLLM_LOG_WARNING("Invalid UCX interface specified: %s will use default interface", userUCXInterface.c_str());
//     }
//     std::map<int, NetINfoT> netInfoSortedMap;
//     for (auto&& [key, netInfo] : mNetInfoMap)
//     {
//         netInfoSortedMap[netInfo.idx] = netInfo;
//     }

//     selectedIp = netInfoSortedMap[0].ipv4;
//     if (selectedIp.empty())
//     {
//         selectedIp = netInfoSortedMap[0].ipv6;
//     }
//     TLLM_LOG_INFO("UCX listener started on interface:%s address: %s:%u", netInfoSortedMap[0].interface.c_str(),
//         selectedIp.c_str(), mListener->getPort());
//     freeifaddrs(ifaddr);
// #endif
// }

// UcxClient::UcxClient()
// {
//     mContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
//     mWorker = mContext->createWorker();
//     int device;
//     TLLM_CUDA_CHECK(cudaGetDevice(&device));
//     mWorker->setProgressThreadStartCallback([device](void* arg) { TLLM_CUDA_CHECK(cudaSetDevice(device)); },
//     nullptr); mWorker->startProgressThread(); initSelfIps();
// }

// void UcxClient::connect(std::string const& ip, std::uint16_t port)
// {
/*
if (mSelfIps.find(ip) != mSelfIps.end())
{
    static const std::string localHost{"127.0.0.1"};
    TLLM_CHECK(mManager.insert(
        "default", std::make_unique<UcxConnection>(mWorker->createEndpointFromHostname(localHost, port))));
}
TLLM_CHECK(
    mManager.insert("default", std::make_unique<UcxConnection>(mWorker->createEndpointFromHostname(ip, port))));
*/
// }

// void UcxClient::initSelfIps()
// {
// #if __linux__
//     struct ifaddrs *ifa, *ifaddr;
//     void* tmpAddrPtr;

//     TLLM_CHECK_WITH_INFO(getifaddrs(&ifaddr) == 0, " UCX initSelfIps getifaddrs call failed\n");
//     TLLM_CHECK_WITH_INFO((ifaddr != NULL), "UCX initSelfIps getifaddrs call failed\n");

//     for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
//     {
//         if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET)
//         {
//             tmpAddrPtr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
//             char buffer[INET_ADDRSTRLEN];
//             inet_ntop(AF_INET, tmpAddrPtr, buffer, INET_ADDRSTRLEN);

//             mSelfIps.insert(std::string(buffer));
//         }
//         else if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET6)
//         {
//             tmpAddrPtr = &((struct sockaddr_in6*) ifa->ifa_addr)->sin6_addr;
//             char buffer[INET6_ADDRSTRLEN];
//             inet_ntop(AF_INET6, tmpAddrPtr, buffer, INET6_ADDRSTRLEN);
//             mSelfIps.insert(std::string(buffer));
//         }
//     }
//     freeifaddrs(ifaddr);
// #endif
// }

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

std::unique_ptr<ConnectionManager> makeUcxConnectionManager(mpi::MpiComm const* comm)
{
    return nullptr;
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace tensorrt_llm::executor::kv_cache

#endif
