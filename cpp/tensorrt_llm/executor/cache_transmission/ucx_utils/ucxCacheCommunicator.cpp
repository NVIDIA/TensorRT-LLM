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
    char ipStr[INET6_ADDRSTRLEN];
    char portStr[INET6_ADDRSTRLEN];
    ucp_conn_request_attr_t attr{};
    UcxConnectionManager* connectionManager = reinterpret_cast<UcxConnectionManager*>(arg);

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    ucxx::utils::ucsErrorThrow(ucp_conn_request_query(connRequest, &attr));
    ucxx::utils::sockaddr_get_ip_port_str(&attr.client_address, ipStr, portStr, INET6_ADDRSTRLEN);
    TLLM_LOG_DEBUG("Server received a connection request from client at address %s:%d ", ipStr, portStr);
    std::cout << "Server received a connection request from client at address " << ipStr << ":" << portStr << std::endl;
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
                std::cout << "InfiniBand Interface: " << ifa->ifa_name << " IP Address: " << address_buffer
                          << std::endl;
                ip = std::string(address_buffer);
                break;
            }
        }
    }

    freeifaddrs(ifaddr);
    return ip;
}

UcxConnectionManager::UcxConnectionManager(tensorrt_llm::mpi::MpiComm const* comm)
    : mComm(comm)
{
    try
    {
        TLLM_CHECK(mComm);
        int device;
        TLLM_CUDA_CHECK(cudaGetDevice(&device));
        mUcxCtx = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);

        try
        {
            mWorkersPool.push_back(mUcxCtx->createWorker());
            mWorkersPool.back().get()->startProgressThread(true);
        }
        catch (std::exception const& e)
        {
            std::string error = "Error creating worker and starting progress thread for rank "
                + std::to_string(mComm->getRank()) + ": " + std::string(e.what());
            throw std::runtime_error(error);
        }

        try
        {
            mListener = mWorkersPool.front()->createListener(listenerPort + comm->getRank(), listenerCallback, this);
        }
        catch (std::exception const& e)
        {
            std::string error = "Error creating listener for rank " + std::to_string(comm->getRank()) + ": " + e.what();
            throw std::runtime_error(error);
        }

        // Get local IP address
        std::string localIp = getLocalIp();
        std::vector<char> localIpBuffer(MAX_IP_LENGTH, 0);
        std::strncpy(localIpBuffer.data(), localIp.c_str(), MAX_IP_LENGTH - 1);

        // Allocate buffer for all IP addresses
        std::vector<char> allIps(comm->getSize() * MAX_IP_LENGTH);

        // Perform Allgather operation
        std::cout << "rank " << comm->getRank() << " | allgather start"
                  << " | comm size " << comm->getSize() << std::endl;
        mComm->allgather(localIpBuffer.data(), allIps.data(), MAX_IP_LENGTH, tensorrt_llm::mpi::MpiType::kCHAR);
        std::cout << "rank " << comm->getRank() << " | allgather done" << std::endl;

        for (int i = 0; i < comm->getSize(); i++)
        {
            std::string ip(allIps.data() + i * MAX_IP_LENGTH);
            std::cout << "rank " << comm->getRank() << " | got IP " << ip << " from rank " << i << std::endl;
            mIpToMpiRank[ip] = i;
        }
        comm->barrier();

        for (int i = comm->getRank() + 1; i < comm->getSize(); i++)
        {
            std::string ip(allIps.data() + i * MAX_IP_LENGTH);
            std::cout << "rank " << comm->getRank() << " | adding connection to rank " << i << " | ip: " << ip
                      << " | port: " << listenerPort + i << std::endl;
            addConnection(ip, listenerPort + i);
        }
    }
    catch (std::exception const& e)
    {
        std::string error = "Error in UcxConnectionManager initialization for rank " + std::to_string(comm->getRank())
            + ": " + e.what();
        throw std::runtime_error(error);
    }
}

void UcxConnectionManager::addConnection(ucp_conn_request_h connRequest)
{
    try
    {
        std::shared_ptr<ucxx::Endpoint> newEp = mListener->createEndpointFromConnRequest(connRequest, true);
        uint64_t id = getNewConnectionId(newEp);
        mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp));
    }
    catch (std::exception const& e)
    {
        std::string error
            = "Error in addConnection(connRequest) for rank " + std::to_string(mComm->getRank()) + ": " + e.what();
        throw std::runtime_error(error);
    }
}

void UcxConnectionManager::addConnection(std::string ip, uint16_t port)
{
    try
    {
        std::shared_ptr<ucxx::Endpoint> newEp = mWorkersPool.front()->createEndpointFromHostname(ip, port, true);
        std::cout << "rank " << mComm->getRank() << " passed createEndpointFromHostname | addConnection | ip: " << ip
                  << " | port: " << port << std::endl;
        uint64_t id = getNewConnectionId(newEp);
        std::cout << "rank " << mComm->getRank() << " passed getNewConnectionId | addConnection | id: " << id
                  << std::endl;
        mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp));
        std::cout << "rank " << mComm->getRank() << " passed emplace | addConnection | id: " << id << std::endl;
    }
    catch (std::exception const& e)
    {
        std::string error = "Error in addConnection(ip) for rank " + std::to_string(mComm->getRank()) + ": " + e.what();
        throw std::runtime_error(error);
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
    while (ucp_ep_query(newEp->getHandle(), &ep_attr) != UCS_OK)
        ;

    ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.remote_sockaddr, rIpStr, portStr, INET6_ADDRSTRLEN);
    std::cout << "rank " << mComm->getRank()
              << " passed sockaddr_get_ip_port_str | getNewConnectionId | rIpStr: " << rIpStr
              << " | portStr: " << portStr << std::endl;
    remotePort = static_cast<ucxx::Tag>(std::stoull(portStr));
    std::cout << "rank " << mComm->getRank() << " passed stoull | getNewConnectionId | remotePort: " << remotePort
              << std::endl;
    remoteIp = std::stoull(rIpStr);
    std::cout << "rank " << mComm->getRank()
              << " passed sockaddr_get_ip_port_str | getNewConnectionId | portStr: " << portStr << std::endl;
    ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.local_sockaddr, lIpStr, portStr, INET6_ADDRSTRLEN);
    localPort = static_cast<ucxx::Tag>(std::stoull(portStr));
    std::cout << "rank " << mComm->getRank() << " passed stoull | getNewConnectionId | lIpStr: " << lIpStr
              << "| localPort: " << localPort << std::endl;
    mMpiRankToConnectionId[mIpToMpiRank[std::string(rIpStr)]]
        = ((remotePort << (32 + 16)) | (localPort << 32) | remoteIp);
    std::cout << "rank " << mComm->getRank()
              << " passed mMpiRankToConnectionId | getNewConnectionId | mMpiRankToConnectionId: "
              << mMpiRankToConnectionId[mIpToMpiRank[std::string(rIpStr)]] << std::endl;
    return ((remotePort << (32 + 16)) | (localPort << 32) | remoteIp);
}

void UcxConnectionManager::initializeConnections()
{
    for (auto& connection : mConnections)
    {
        std::cout << "UcxConnectionManager::initializeConnections " << std::endl;
        connection.second->initialize(shared_from_this());
    }
}

Connection const* UcxConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    uint64_t senderTag;
    ucp_request_param_t tagRecvParams;
    tagRecvParams.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    tagRecvParams.cb.recv = [](void* request, ucs_status_t status, ucp_tag_recv_info_t const* info,
                                void* userData) -> void { *(uint64_t*) userData = info->sender_tag; };
    tagRecvParams.user_data = &senderTag;

    auto request = ucp_tag_recv_nbx(mWorkersPool.front().get()->getHandle(), data, size, 1, 0, &tagRecvParams);

    while (ucp_request_check_status(request) != UCS_INPROGRESS)
        ;

    return mConnections[senderTag].get();
}

std::vector<Connection const*> UcxConnectionManager::getConnections(CommState const& state)
{
    std::vector<Connection const*> ret;
    TLLM_CHECK(state.isMpiState());
    for (auto rank : state.getMpiState().mRanks)
    {
        ret.emplace_back(mConnections[mMpiRankToConnectionId[rank]].get());
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
        throw std::runtime_error(error);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace tensorrt_llm::executor::kv_cache
