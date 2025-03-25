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

#include "tensorrt_llm/executor/cache_transmission/ucx_utils/ucxCacheCommunicator.h"
#include "tensorrt_llm/common/logger.h"
#include <exception>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

const uint16_t listenerPort = 12345;
int const MAX_IP_LENGTH = 16;
static void listenerCallback(ucp_conn_request_h connRequest, void* arg);
static std::string getLocalIp();

static void listenerCallback(ucp_conn_request_h connRequest, void* arg)
{
    TLLM_LOG_DEBUG("listenerCallback");
    char ipStr[INET6_ADDRSTRLEN];
    char portStr[INET6_ADDRSTRLEN];
    ucp_conn_request_attr_t attr{};
    UcxConnectionManager* connectionManager = reinterpret_cast<UcxConnectionManager*>(arg);

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    ucxx::utils::ucsErrorThrow(ucp_conn_request_query(connRequest, &attr));
    ucxx::utils::sockaddr_get_ip_port_str(&attr.client_address, ipStr, portStr, INET6_ADDRSTRLEN);
    TLLM_LOG_DEBUG("Server received a connection request from client at address %s:%s", ipStr, portStr);
    connectionManager->addConnection(connRequest);
}

static std::string getLocalIp()
{
    struct ifaddrs *ifaddr, *ifa;
    void* addr_ptr;
    std::string ip("UNKNOWN IP");

    // Get the list of network interfaces
    if (getifaddrs(&ifaddr) == -1)
    {
        perror("getifaddrs");
        return ip;
    }

    // Loop through the linked list of interfaces
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
    {
        // Check if the interface is an IP interface
        if (ifa->ifa_addr == nullptr)
            continue;

        // Check for InfiniBand interfaces (usually named ib0, ib1, etc.)
        if (std::string(ifa->ifa_name).find("ib") == 0)
        {
            // Check if the address family is AF_INET (IPv4)
            if (ifa->ifa_addr->sa_family == AF_INET)
            {
                addr_ptr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
                char address_buffer[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, addr_ptr, address_buffer, sizeof(address_buffer));
                TLLM_LOG_DEBUG("InfiniBand Interface: %s IP Address: %s", ifa->ifa_name, address_buffer);
                ip = std::string(address_buffer);
                break;
            }
        }
    }

    freeifaddrs(ifaddr);
    return ip;
}

uint64_t UcxConnectionManager::getLocalGID() const
{
    if (mComm)
    {
        return mComm->getRank();
    }
    return std::numeric_limits<uint64_t>::max();
}

UcxConnectionManager::UcxConnectionManager(tensorrt_llm::mpi::MpiComm const* comm)
    : mComm(comm)
{
    try
    {
        TLLM_CHECK(mComm);
        int device;
        TLLM_CUDA_CHECK(cudaGetDevice(&device));
        TLLM_LOG_DEBUG("UcxConnectionManager | rank %d | device: %d", mComm->getRank(), device);
        mUcxCtx = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);

        try
        {
            mWorkersPool.push_back(mUcxCtx->createWorker());
            mWorkersPool.back().get()->setProgressThreadStartCallback(
                [device](void* arg) { TLLM_CUDA_CHECK(cudaSetDevice(device)); }, nullptr);
            mWorkersPool.back().get()->startProgressThread(true);
        }
        catch (std::exception const& e)
        {
            std::string error = "Error creating worker and starting progress thread for rank "
                + std::to_string(mComm->getRank()) + ": " + std::string(e.what());
            TLLM_THROW(error);
        }

        try
        {
            TLLM_LOG_DEBUG("rank %d | creating listener on port %d", mComm->getRank(), listenerPort + mComm->getRank());
            mListener = mWorkersPool.front()->createListener(listenerPort + mComm->getRank(), listenerCallback, this);
        }
        catch (std::exception const& e)
        {
            std::string error = "Error creating listener for rank " + std::to_string(comm->getRank()) + ": " + e.what();
            TLLM_THROW(error);
        }

        // Get local IP address
        std::string localIp = getLocalIp();
        std::vector<char> localIpBuffer(MAX_IP_LENGTH, 0);
        std::strncpy(localIpBuffer.data(), localIp.c_str(), MAX_IP_LENGTH - 1);

        // Allocate buffer for all IP addresses
        std::vector<char> allIps(comm->getSize() * MAX_IP_LENGTH);
        TLLM_LOG_DEBUG("rank %d | comm split start - barrier| world comm size %d", comm->getRank(), comm->getSize());
        comm->barrier();
        TLLM_LOG_DEBUG(
            "rank %d | comm split start - barrier passed | world comm size %d", comm->getRank(), comm->getSize());
        // auto worldComm = std::addressof(tensorrt_llm::mpi::MpiComm::session());
        tensorrt_llm::mpi::MpiComm kvCacheComm(comm->split(300, mComm->getRank()));
        TLLM_LOG_DEBUG("rank %d | comm split done | kvCacheComm size %d, kvCacheComm rank %d", comm->getRank(),
            kvCacheComm.getSize(), kvCacheComm.getRank());

        int excludeRank0 = comm->getSize() != kvCacheComm.getSize();

        // Perform Allgather operation
        TLLM_LOG_DEBUG("rank %d | allgather start | world comm size %d,", comm->getRank(), comm->getSize());
        kvCacheComm.allgather(localIpBuffer.data(), allIps.data(), MAX_IP_LENGTH, tensorrt_llm::mpi::MpiType::kCHAR);
        TLLM_LOG_DEBUG("rank %d | allgather done | kvCacheComm size %d, kvCacheComm rank %d", comm->getRank(),
            kvCacheComm.getSize(), kvCacheComm.getRank());

        for (int i = kvCacheComm.getRank() + 1; i < kvCacheComm.getSize(); i++)
        {
            int globalRank = i + (excludeRank0 ? 1 : 0);
            std::string ip(allIps.data() + i * MAX_IP_LENGTH);
            TLLM_LOG_DEBUG("kvCacheComm rank %d | adding connection to kvCacheComm rank %d | ip: %s | port: %d",
                kvCacheComm.getRank(), i, ip.c_str(), listenerPort + globalRank);
            mGIDToConnectionId[globalRank] = addConnection(ip, listenerPort + globalRank);
            mConnections[mGIDToConnectionId[globalRank]].get()->mRemoteGID = globalRank;
            mConnections[mGIDToConnectionId[globalRank]].get()->sendGID();
        }
        TLLM_LOG_DEBUG("rank %d | done initiating connections", comm->getRank());

        while (static_cast<int>(mGIDToConnectionId.size()) < kvCacheComm.getSize() - 1)
        {
            TLLM_LOG_DEBUG("kvCacheComm rank %d | waiting for %d connections, current size %d", comm->getRank(),
                kvCacheComm.getSize(), mConnections.size());
            std::unique_lock<std::mutex> lock(mPendingGIDFuturesMutex);
            if (!mPendingGIDFutures.empty())
            {
                std::shared_ptr<std::future<void>> future = mPendingGIDFutures.front();
                mPendingGIDFutures.pop();
                lock.unlock();
                future->get();
            }
            else
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        TLLM_LOG_DEBUG("rank %d | done initiating connections | mGIDToConnectionId size %d", comm->getRank(),
            mGIDToConnectionId.size());
    }
    catch (std::exception const& e)
    {
        std::string error = "Error in UcxConnectionManager initialization for rank " + std::to_string(comm->getRank())
            + ": " + e.what();
        TLLM_THROW(error);
    }
}

UcxConnectionManager::~UcxConnectionManager()
{
    for (auto& worker : mWorkersPool)
    {
        worker->stopProgressThread();
    }
}

void UcxConnectionManager::updateGIDToConnectionIdMap(
    std::shared_ptr<ucxx::Request> request, uint64_t* gid, uint64_t connectionId)
{
    while (request->isCompleted() == false)
        ;
    std::unique_lock<std::mutex> lock(mGIDToConnectionIdMutex);
    mGIDToConnectionId[*gid] = connectionId;
    lock.unlock();
    TLLM_LOG_DEBUG("updateGIDToConnectionIdMap | rank %d | remote gid: %lu | connectionId: %lu", mComm->getRank(), *gid,
        connectionId);
}

uint64_t UcxConnectionManager::addConnection(ucp_conn_request_h connRequest)
{
    try
    {
        std::shared_ptr<ucxx::Endpoint> newEp = mListener->createEndpointFromConnRequest(connRequest, true);
        uint64_t id = getNewConnectionId(newEp);
        TLLM_LOG_DEBUG("addConnection | rank %d | emplacing connection id %lu", mComm->getRank(), id);
        mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp, this));
        std::shared_ptr<ucxx::Request> request = newEp->streamRecv(
            reinterpret_cast<void*>(&mConnections[id]->mRemoteGID), sizeof(mConnections[id]->mRemoteGID), false);
        std::unique_lock<std::mutex> lock(mPendingGIDFuturesMutex);
        mPendingGIDFutures.push(std::make_shared<std::future<void>>(std::async(std::launch::async,
            [this, request, id]() { this->updateGIDToConnectionIdMap(request, &mConnections[id]->mRemoteGID, id); })));
        lock.unlock();
        return id;
    }
    catch (std::exception const& e)
    {
        std::string error
            = "Error in addConnection(connRequest) for rank " + std::to_string(mComm->getRank()) + ": " + e.what();
        TLLM_THROW(error);
    }
}

uint64_t UcxConnectionManager::addConnection(std::string ip, uint16_t port)
{
    try
    {
        std::shared_ptr<ucxx::Endpoint> newEp = mWorkersPool.front()->createEndpointFromHostname(ip, port, true);
        TLLM_LOG_DEBUG("rank %d passed createEndpointFromHostname | addConnection | ip: %s | port: %d",
            mComm->getRank(), ip.c_str(), port);
        uint64_t id = getNewConnectionId(newEp);
        TLLM_LOG_DEBUG("rank %d passed getNewConnectionId | addConnection | id: %lu", mComm->getRank(), id);
        mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp, this));
        TLLM_LOG_DEBUG("rank %d passed emplace | addConnection | id: %lu", mComm->getRank(), id);
        return id;
    }
    catch (std::exception const& e)
    {
        std::string error = "Error in addConnection(ip) for rank " + std::to_string(mComm->getRank()) + ": " + e.what();
        TLLM_THROW(error);
    }
}

uint64_t UcxConnectionManager::getNewConnectionId(std::shared_ptr<ucxx::Endpoint> newEp)
{
    ucp_ep_attr_t ep_attr;
    ep_attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR | UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;
    uint64_t remotePort, localPort;
    uint32_t remoteIp;
    char lIpStr[INET6_ADDRSTRLEN];
    char rIpStr[INET6_ADDRSTRLEN];
    char portStr[INET6_ADDRSTRLEN];
    // ucs_status_t status = ucp_ep_query(newEp->getHandle(), &ep_attr);
    ucs_status_t status;
    while ((status = ucp_ep_query(newEp->getHandle(), &ep_attr)) != UCS_OK)
    {
        TLLM_LOG_DEBUG("rank %d | ucp_ep_query failed | retrying | status: %d", mComm->getRank(), status);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.remote_sockaddr, rIpStr, portStr, INET6_ADDRSTRLEN);
    TLLM_LOG_DEBUG("rank %d passed sockaddr_get_ip_port_str | getNewConnectionId | rIpStr: %s | portStr: %s",
        mComm->getRank(), rIpStr, portStr);
    remotePort = static_cast<ucxx::Tag>(std::stoull(portStr));
    TLLM_LOG_DEBUG("rank %d passed stoull | getNewConnectionId | remotePort: %lu", mComm->getRank(), remotePort);
    remoteIp = std::stoull(rIpStr);
    TLLM_LOG_DEBUG(
        "rank %d passed sockaddr_get_ip_port_str | getNewConnectionId | portStr: %s", mComm->getRank(), portStr);
    ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.local_sockaddr, lIpStr, portStr, INET6_ADDRSTRLEN);
    localPort = static_cast<ucxx::Tag>(std::stoull(portStr));
    TLLM_LOG_DEBUG("rank %d passed stoull | getNewConnectionId | lIpStr: %s | localPort: %lu", mComm->getRank(), lIpStr,
        localPort);
    return ((remotePort << (32 + 16)) | (localPort << 32) | remoteIp);
}

Connection const* UcxConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    uint64_t senderTag;
    ucp_request_param_t tagRecvParams;
    tagRecvParams.op_attr_mask
        = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    tagRecvParams.cb.recv
        = [](void* request, ucs_status_t status, ucp_tag_recv_info_t const* info, void* userData) -> void
    {
        *(uint64_t*) userData = info->sender_tag;
        TLLM_LOG_DEBUG("recvConnect | senderTag: %lu", *(uint64_t*) userData);
    };
    tagRecvParams.user_data = &senderTag;

    auto request = ucp_tag_recv_nbx(mWorkersPool.front().get()->getHandle(), data, size, 1, 0, &tagRecvParams);
    ucs_status_t status;
    while ((status = ucp_request_check_status(request)) == UCS_INPROGRESS)
        ;
    if (request)
    {
        ucp_request_free(request);
    }
    if (status != UCS_OK)
    {
        TLLM_THROW("Error in recvConnect" + std::to_string(status));
    }
    TLLM_LOG_DEBUG("recvConnect2 | rank %d | senderTag: %lu", getLocalGID(), senderTag);

    return mConnections[senderTag].get();
}

std::vector<Connection const*> UcxConnectionManager::getConnections(CommState const& state)
{
    std::vector<Connection const*> ret;
    TLLM_CHECK(state.isMpiState());
    for (auto rank : state.getMpiState().mRanks)
    {
        TLLM_LOG_DEBUG("local gid: %lu | getConnections | emplacing connection with remote gid %d, connection id %lu",
            getLocalGID(), rank, mGIDToConnectionId[rank]);
        ret.emplace_back(mConnections[mGIDToConnectionId[rank]].get());
    }
    return ret;
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

std::unique_ptr<ConnectionManager> makeMpiConnectionManager(mpi::MpiComm const* comm)
return UcxConnectionManager::create(comm);
    }
    catch (std::exception const& e)
    {
        std::string error
            = "Error in makeUcxConnectionManager for rank " + std::to_string(comm->getRank()) + ": " + e.what();
        TLLM_THROW(error);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace tensorrt_llm::executor::kv_cache
