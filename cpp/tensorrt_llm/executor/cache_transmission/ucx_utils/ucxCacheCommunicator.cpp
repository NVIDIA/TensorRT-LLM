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
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    struct hostent* hostEntry = gethostbyname(hostname);
    if (hostEntry != nullptr)
    {
        return inet_ntoa(*((struct in_addr*) hostEntry->h_addr_list[0]));
    }
    return "Unknown";
}

UcxConnectionManager::UcxConnectionManager(mpi::MpiComm const* comm)
    : mComm(comm)
{
    try
    {
        TLLM_CHECK(mComm);
        int device;
        TLLM_CUDA_CHECK(cudaGetDevice(&device));
        mUcxCtx = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);

        mWorkersPool.push_back(mUcxCtx->createWorker());
        mWorkersPool.back().get()->startProgressThread(true);

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
        mComm->allgather(
            localIpBuffer.data(), allIps.data(), comm->getSize() * MAX_IP_LENGTH, tensorrt_llm::mpi::MpiType::kCHAR);

        for (int i = 0; i < comm->getSize(); i++)
        {
            std::string ip(allIps.data() + i * MAX_IP_LENGTH);
            TLLM_LOG_DEBUG("rank %d | got IP %s from rank %d", comm->getRank(), ip.c_str(), i);
            mIpToMpiRank[ip] = i;
        }
        comm->barrier();

        for (int i = comm->getRank() + 1; i < comm->getSize(); i++)
        {
            std::string ip(allIps.data() + i * MAX_IP_LENGTH);
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
        mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp, shared_from_this()));
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
        uint64_t id = getNewConnectionId(newEp);
        mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp, shared_from_this()));
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
    ucs_status_t status = ucp_ep_query(newEp->getHandle(), &ep_attr);
    if (status == UCS_OK)
    {
        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.remote_sockaddr, rIpStr, portStr, INET6_ADDRSTRLEN);
        remotePort = static_cast<ucxx::Tag>(std::stoull(portStr));
        remoteIp = std::stoull(lIpStr);
        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.local_sockaddr, lIpStr, portStr, INET6_ADDRSTRLEN);
        localPort = static_cast<ucxx::Tag>(std::stoull(portStr));
        mMpiRankToConnectionId[mIpToMpiRank[std::string(rIpStr)]]
            = ((remotePort << (32 + 16)) | (localPort << 32) | remoteIp);
        return ((remotePort << (32 + 16)) | (localPort << 32) | remoteIp);
    }
    else
    {
        if (status == UCS_ERR_NOT_CONNECTED)
        {
            TLLM_LOG_ERROR("UCX connection has not been established yet");
        }
    }
    return 0;
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
{
    try
    {
        return std::make_unique<UcxConnectionManager>(comm);
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
