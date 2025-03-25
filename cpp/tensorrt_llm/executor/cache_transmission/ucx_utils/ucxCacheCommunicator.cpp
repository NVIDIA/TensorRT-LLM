#include "tensorrt_llm/executor/cache_transmission/ucx_utils/ucxCacheCommunicator.h"
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

const uint16_t listener_port = 12345;
int const MAX_IP_LENGTH = 16;
static void listener_cb(ucp_conn_request_h conn_request, void* arg);
static std::string getLocalIP();

static void listener_cb(ucp_conn_request_h conn_request, void* arg)
{
    char ip_str[INET6_ADDRSTRLEN];
    char port_str[INET6_ADDRSTRLEN];
    ucp_conn_request_attr_t attr{};
    UcxConnectionManager* connectionManager = reinterpret_cast<UcxConnectionManager*>(arg);

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    ucxx::utils::ucsErrorThrow(ucp_conn_request_query(conn_request, &attr));
    ucxx::utils::sockaddr_get_ip_port_str(&attr.client_address, ip_str, port_str, INET6_ADDRSTRLEN);
    TLLM_LOG_DEBUG("Server received a connection request from client at address %s:%d ", ip_str, port_str);
    connectionManager->addConnection(conn_request);
}

static std::string getLocalIP()
{
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    struct hostent* host_entry = gethostbyname(hostname);
    if (host_entry != nullptr)
    {
        return inet_ntoa(*((struct in_addr*) host_entry->h_addr_list[0]));
    }
    return "Unknown";
}

UcxConnectionManager::UcxConnectionManager(mpi::MpiComm const* comm)
    : mComm(comm)
{
    TLLM_CHECK(mComm);
    int device;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    mUcxCtx = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    mWorkersPool.push_back(mUcxCtx->createWorker()); // TODO: support configurable workers pool size
    mWorkersPool.back().get()->startProgressThread(true);
    mListener = mWorkersPool.front()->createListener(listener_port, listener_cb, this);
    // Get local IP address
    std::string localIP = getLocalIP();
    std::vector<char> localIPBuffer(MAX_IP_LENGTH, 0);
    std::strncpy(localIPBuffer.data(), localIP.c_str(), MAX_IP_LENGTH - 1);

    // Allocate buffer for all IP addresses
    std::vector<char> allIPs(comm->getSize() * MAX_IP_LENGTH);

    // Perform Allgather operation
    mComm->allgather(
        localIPBuffer.data(), allIPs.data(), comm->getSize() * MAX_IP_LENGTH, tensorrt_llm::mpi::MpiType::kCHAR);
    for (int i = comm->getRank() + 1; i < comm->getSize(); i++)
    {
        std::string Ip(allIPs.data() + i * MAX_IP_LENGTH);
        TLLM_LOG_DEBUG("rank %d | got IP %s from rank %d", comm->getRank(), Ip.c_str(), i);
        addConnection(Ip);
    }
}

void UcxConnectionManager::addConnection(ucp_conn_request_h conn_request)
{
    std::shared_ptr<ucxx::Endpoint> newEp = mListener->createEndpointFromConnRequest(conn_request, true);
    uint64_t id = getNewConnectionId(newEp);
    mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp, shared_from_this()));
}

void UcxConnectionManager::addConnection(std::string Ip)
{
    std::shared_ptr<ucxx::Endpoint> newEp
        = mWorkersPool.front()->createEndpointFromHostname(Ip, listener_port, true); // TODO: workers round robin
    uint64_t id = getNewConnectionId(newEp);
    mConnections.emplace(id, std::make_shared<UcxConnection>(id, newEp, shared_from_this()));
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

std::unique_ptr<ConnectionManager> makeMpiConnectionManager(mpi::MpiComm const* comm)
{
    return std::make_unique<UcxConnectionManager>(comm);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace tensorrt_llm::executor::kv_cache
