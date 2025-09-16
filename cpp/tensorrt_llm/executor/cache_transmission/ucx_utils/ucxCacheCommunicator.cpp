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
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include <exception>
#include <iostream>
#include <mutex>
#include <regex>
#include <sys/socket.h>
#include <ucxx/address.h>
#include <ucxx/typedefs.h>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

class UcxCmMessage
{
public:
    enum class MessageType
    {
        GET_WORKER_ADDRESS = 1,
        SERVER_WORKER_ADDRESS = 2,
        STOP = 3,
    };

    MessageType mType;
    std::optional<std::string> mWorkerAddress;

    UcxCmMessage(MessageType type, std::optional<std::string> workerAddress)
        : mType(type)
        , mWorkerAddress(std::move(workerAddress))
    {
    }

    static size_t serializedSize(UcxCmMessage const& message)
    {
        namespace su = tensorrt_llm::executor::serialize_utils;

        return su::serializedSize(message.mType) + su::serializedSize(message.mWorkerAddress);
    }

    static void serialize(UcxCmMessage const& message, std::ostream& os)
    {
        namespace su = tensorrt_llm::executor::serialize_utils;
        su::serialize(message.mType, os);
        su::serialize(message.mWorkerAddress, os);
    }

    static UcxCmMessage deserialize(std::istream& is)
    {
        namespace su = tensorrt_llm::executor::serialize_utils;
        auto type = su::deserialize<MessageType>(is);
        auto workerAddress = su::deserialize<std::optional<std::string>>(is);
        return UcxCmMessage(type, workerAddress);
    }
};

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
    bool getIp = false;
    // Loop through the linked list of interfaces
    for (int ipVIndex = 0; ipVIndex < 2; ipVIndex++)
    {
        for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
        {
            // Check if the interface is an IP interface
            if (ifa->ifa_addr == nullptr)
                continue;

            std::string ucxInterface = common::getEnvUCXInterface();
            if (!ucxInterface.empty() && strcmp(ifa->ifa_name, ucxInterface.c_str()) != 0)
            {
                continue;
            }

            // Skip the loopback interface
            if (ucxInterface.empty() && (strncmp(ifa->ifa_name, "docker", 6) == 0 || strcmp(ifa->ifa_name, "lo") == 0))
            {
                continue;
            }

            // Check if the address family is AF_INET (IPv4)
            int addressFamily = (ipVIndex == 0) ? AF_INET : AF_INET6;

            if (ifa->ifa_addr->sa_family == addressFamily)
            {
                if (ipVIndex == 0) // IPv4
                {
                    addr_ptr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
                    char address_buffer[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, addr_ptr, address_buffer, sizeof(address_buffer));

                    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " ***** UCX    Interface: %s IP Address: %s",
                        ifa->ifa_name, address_buffer);
                    ip = address_buffer;
                }
                else // IPv6
                {
                    addr_ptr = &((struct sockaddr_in6*) ifa->ifa_addr)->sin6_addr;
                    char address_buffer[INET6_ADDRSTRLEN];
                    inet_ntop(AF_INET6, addr_ptr, address_buffer, sizeof(address_buffer));

                    if (address_buffer[0] == 'f' && address_buffer[1] == 'e' && address_buffer[2] == '8'
                        && address_buffer[3] == '0')
                    {
                        // local link address

                        ip = std::string(address_buffer) + "%" + ifa->ifa_name;
                    }
                    else
                    {
                        ip = address_buffer;
                    }
                    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " ***** UCX    Interface: %s IP Address: %s",
                        ifa->ifa_name, ip.c_str());
                }

                getIp = true;
                break;
            }
        }
        if (getIp)
        {
            break;
        }
    }
    if (!getIp)
    {
        TLLM_LOG_ERROR(mpi::MpiComm::world().getRank(),
            "UCX   No valid IP address found please set correct UCX interface with env variable TRTLLM_UCX_INTERFACE");
    }

    freeifaddrs(ifaddr);
    return ip;
}

std::optional<std::pair<std::string, int>> parse_zmq_endpoint(std::string const& endpoint)
{
    std::regex ipv4_regex(R"(tcp://([\d\.]+):(\d+))");
    std::regex ipv6_regex(R"(tcp://\[([0-9a-fA-F:%\w]+)\]:(\d+))");
    std::smatch match;
    if (std::regex_match(endpoint, match, ipv4_regex))
    {
        return std::make_pair(match[1].str(), std::stoi(match[2].str()));
    }
    else if (std::regex_match(endpoint, match, ipv6_regex))
    {
        return std::make_pair(match[1].str(), std::stoi(match[2].str()));
    }
    return std::nullopt;
}

UcxConnectionManager::UcxConnectionManager()

{
    try
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&mDevice));
        mUcxCtx = ucxx::createContext({{"RNDV_PIPELINE_ERROR_HANDLING", "y"}}, UCP_FEATURE_TAG);
        int device = mDevice;
        try
        {
            mWorkersPool.push_back(mUcxCtx->createWorker());
            mWorkersPool.back().get()->setProgressThreadStartCallback(
                [device](void* arg) { TLLM_CUDA_CHECK(cudaSetDevice(device)); }, nullptr);
            mWorkersPool.back().get()->startProgressThread(true);
        }
        catch (std::exception const& e)
        {
            std::string error = "Error creating worker and starting progress thread for rank " + std::string(e.what());
            TLLM_THROW(error);
        }
        auto workerAddressPtr = mWorkersPool.front()->getAddress();
        mWorkerAddress = workerAddressPtr->getString();

        mZmqRepSocket = zmq::socket_t(mZmqContext, zmq::socket_type::rep);
        mZmqRepSocket.set(zmq::sockopt::sndhwm, 1000);
        std::string localIp = getLocalIp();
        if (localIp.find(':') != std::string::npos)
        {
            // ipv6
            mZmqRepSocket.set(zmq::sockopt::ipv6, 1);

            localIp = "[" + localIp + "]";
        }
        TLLM_LOG_INFO(
            mpi::MpiComm::world().getRank(), "UcxConnectionManager::UcxConnectionManager localIp: %s", localIp.c_str());
        mZmqRepSocket.bind("tcp://" + localIp + ":*");
        mZmqRepEndpoint = mZmqRepSocket.get(zmq::sockopt::last_endpoint);
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(), "UcxConnectionManager::UcxConnectionManager mZmqRepEndpoint: %s",
            mZmqRepEndpoint.c_str());
        auto parse_result = parse_zmq_endpoint(mZmqRepEndpoint);
        TLLM_CHECK_WITH_INFO(parse_result.has_value(), "Failed to parse ZMQ endpoint");
        auto [ip, port] = parse_result.value();
        TLLM_LOG_INFO(mpi::MpiComm::world().getRank(), "UcxConnectionManager::UcxConnectionManager ip: %s, port: %d",
            ip.c_str(), port);

        SocketState socketState{static_cast<uint16_t>(port), ip};
        std::vector<executor::kv_cache::SocketState> socketStates(mpi::MpiComm::session().getSize());

        if (mpi::MpiComm::session().getSize() > 1)
        {

            mpi::MpiComm::session().barrier();
            namespace su = executor::serialize_utils;

            std::ostringstream oStream;
            su::serialize(socketState, oStream);
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
                socketStates[i] = su::deserialize<executor::kv_cache::SocketState>(is);
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " recv  socketStates[%d]: %s", i,
                    socketStates[i].toString().c_str());
            }
        }
        else
        {
            socketStates[0] = socketState;
        }
        mCommState = CommState(socketStates, mpi::MpiComm::session().getRank());
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " ***** UCX    mCommState: %s", mCommState.toString().c_str());

        mZmqRepThread = std::thread(
            [this]()
            {
                while (true)
                {
                    zmq::message_t message;
                    auto ret = mZmqRepSocket.recv(message);
                    TLLM_CHECK_WITH_INFO(ret, "mZmqRepSocket.recv failed");
                    std::string recvMessage(static_cast<char*>(message.data()), message.size());
                    std::istringstream is(recvMessage);
                    UcxCmMessage ucxCmessage = UcxCmMessage::deserialize(is);

                    if (ucxCmessage.mType == UcxCmMessage::MessageType::GET_WORKER_ADDRESS)
                    {
                        // add Connection
                        TLLM_CHECK_WITH_INFO(ucxCmessage.mWorkerAddress.has_value(), "workerAddress is null");
                        std::string workerAddress = ucxCmessage.mWorkerAddress.value();
                        std::string selfWorkerAddress = mWorkerAddress;
                        UcxCmMessage serverMessage(UcxCmMessage::MessageType::SERVER_WORKER_ADDRESS, selfWorkerAddress);
                        std::ostringstream oStream;
                        UcxCmMessage::serialize(serverMessage, oStream);
                        std::string serverMessageStr = oStream.str();
                        mZmqRepSocket.send(zmq::buffer(serverMessageStr), zmq::send_flags::none);
                        addConnection(workerAddress);
                    }
                    else if (ucxCmessage.mType == UcxCmMessage::MessageType::STOP)
                    {
                        UcxCmMessage stopMessage(UcxCmMessage::MessageType::STOP, std::nullopt);
                        std::ostringstream oStream;
                        UcxCmMessage::serialize(stopMessage, oStream);
                        std::string stopMessageStr = oStream.str();
                        mZmqRepSocket.send(zmq::buffer(stopMessageStr), zmq::send_flags::none);
                        break;
                    }
                    else
                    {
                        TLLM_THROW("Zmq recv unknown message: %s", recvMessage.c_str());
                    }
                }
            });
    }
    catch (std::exception const& e)
    {
        std::string error = std::string("Error in UcxConnectionManager initialization for rank ") + e.what();
        TLLM_THROW(error);
    }
}

UcxConnectionManager::~UcxConnectionManager()
{
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "UcxConnectionManager::~UcxConnectionManager");

    for (auto& worker : mWorkersPool)
    {
        worker->stopProgressThread();
    }
    if (mZmqRepThread.joinable())
    {
        zmq::socket_t socket(mZmqContext, zmq::socket_type::req);
        socket.set(zmq::sockopt::ipv6, 1);
        socket.connect(mZmqRepEndpoint);
        UcxCmMessage stopMessage(UcxCmMessage::MessageType::STOP, std::nullopt);
        std::ostringstream oStream;
        UcxCmMessage::serialize(stopMessage, oStream);
        std::string stopMessageStr = oStream.str();
        socket.send(zmq::buffer(stopMessageStr), zmq::send_flags::none);
        zmq::message_t reply;
        auto ret = socket.recv(reply);
        TLLM_CHECK_WITH_INFO(ret, "zmq socket.recv failed");
        std::string replyStr(static_cast<char*>(reply.data()), reply.size());
        std::istringstream is(replyStr);
        UcxCmMessage serverMessage = UcxCmMessage::deserialize(is);
        TLLM_CHECK_WITH_INFO(serverMessage.mType == UcxCmMessage::MessageType::STOP, "serverMessage.mType is not STOP");
        socket.close();
        mZmqRepThread.join();
    }

    mZmqRepSocket.close();

    mZmqContext.close();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "END UcxConnectionManager::~UcxConnectionManager");
}

void UcxConnectionManager::addConnection(std::string const& workerAddress)
{
    try
    {
        auto workerAddressPtr = ucxx::createAddressFromString(workerAddress);
        auto newEp = mWorkersPool.front()->createEndpointFromWorkerAddress(workerAddressPtr, true);

        UcxConnection::ConnectionIdType connectionId = getNewConnectionId(newEp);
        std::scoped_lock lock(mConnectionFuturesMutex);

        std::future<void> future = std::async(std::launch::async,
            [this, connectionId, newEp]()
            {
                std::scoped_lock lock(mConnectionsMutex);
                std::shared_ptr<UcxConnection> connection
                    = std::make_shared<UcxConnection>(connectionId, newEp, this, false);
                mConnections.emplace(connectionId, connection);
            });
        mConnectionFutures.emplace(connectionId, std::move(future));
    }
    catch (std::exception const& e)
    {
        std::string error = "Error in addConnection(connRequest) for rank "
            + std::to_string(mpi::MpiComm::world().getRank()) + ": " + e.what();
        TLLM_THROW(error);
    }
}

std::string build_zmq_endpoint(std::string const& ip, uint16_t port)
{
    std::ostringstream oss;

    std::regex ipv6_regex(R"([0-9a-fA-F]*:[0-9a-fA-F]*:[0-9a-fA-F]*.*)");
    if (std::regex_match(ip, ipv6_regex) && ip.find(':') != std::string::npos)
    {
        oss << "tcp://[" << ip << "]:" << port;
    }
    else
    {
        oss << "tcp://" << ip << ":" << port;
    }

    return oss.str();
}

UcxConnection::ConnectionIdType UcxConnectionManager::addConnection(std::string const& ip, uint16_t port)
{
    static std::mutex sAddConnectionIPMutex;
    try
    {
        std::shared_ptr<UcxConnection> connection;
        UcxConnection::ConnectionIdType connectionId = 0;
        {
            std::scoped_lock addConnectionIPLock(sAddConnectionIPMutex);
            // This lock ensures that only one thread can create an endpoint from hostname and establish a UCX
            // connection at a time, guaranteeing that the only one listener will send connectionId to requester in the
            // same time.
            auto reqSocket = zmq::socket_t(mZmqContext, zmq::socket_type::req);
            reqSocket.set(zmq::sockopt::ipv6, 1);
            reqSocket.connect(build_zmq_endpoint(ip, port));
            UcxCmMessage getWorkerAddressMessage(UcxCmMessage::MessageType::GET_WORKER_ADDRESS, mWorkerAddress);
            std::ostringstream oStream;
            UcxCmMessage::serialize(getWorkerAddressMessage, oStream);
            std::string getWorkerAddressMessageStr = oStream.str();
            reqSocket.send(zmq::buffer(getWorkerAddressMessageStr), zmq::send_flags::none);
            zmq::message_t reply;
            auto ret = reqSocket.recv(reply);
            TLLM_CHECK_WITH_INFO(ret, "zmq socket.recv failed");
            std::string replyStr(static_cast<char*>(reply.data()), reply.size());
            std::istringstream is(replyStr);
            UcxCmMessage serverMessage = UcxCmMessage::deserialize(is);
            TLLM_CHECK_WITH_INFO(serverMessage.mType == UcxCmMessage::MessageType::SERVER_WORKER_ADDRESS,
                "serverMessage.mType is not SERVER_WORKER_ADDRESS");
            std::string serverWorkerAddress = serverMessage.mWorkerAddress.value();
            auto serverWorkerAddressPtr = ucxx::createAddressFromString(serverWorkerAddress);
            auto newEp = mWorkersPool.front()->createEndpointFromWorkerAddress(serverWorkerAddressPtr, true);
            connectionId = getNewConnectionId(newEp);
            connection = std::make_shared<UcxConnection>(connectionId, newEp, this, true);
        }
        TLLM_CHECK(connectionId != 0);
        std::scoped_lock lock(mConnectionsMutex, mAddressToConnectionIdMutex);
        mConnections.emplace(connectionId, connection);
        std::string address = ip + ":" + std::to_string(port);
        mAddressToConnectionId[address] = connectionId;
        return connectionId;
    }
    catch (std::exception const& e)
    {
        std::string error = "Error in addConnection(ip) for rank " + std::to_string(mpi::MpiComm::world().getRank())
            + " ip: " + ip + " port: " + std::to_string(port) + ": " + e.what();
        TLLM_THROW(error);
    }
}

UcxConnection::ConnectionIdType UcxConnectionManager::getNewConnectionId(std::shared_ptr<ucxx::Endpoint> const& newEp)
{
    return mConnectionIdCounter++;
}

Connection const* UcxConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{
    std::vector<char> buffer(size + sizeof(UcxConnection::ConnectionIdType));
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };

    std::shared_ptr<ucxx::Request> req = mWorkersPool.front()->tagRecv(
        buffer.data(), buffer.size(), ucxx::Tag(ctx.getTag()), ucxx::TagMask(0xFFFFFFFF), false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv SendConnectionId should be completed");
    req->checkError();

    memcpy(data, buffer.data(), size);
    UcxConnection::ConnectionIdType connectionId
        = *reinterpret_cast<UcxConnection::ConnectionIdType*>(buffer.data() + size);
    std::scoped_lock lock(mConnectionsMutex, mConnectionFuturesMutex);
    TLLM_CHECK_WITH_INFO(mConnectionFutures.find(connectionId) != mConnectionFutures.end(),
        "connectionFuture not found In recvConnect connectionId : %lu , worldRank: %d", connectionId,
        mpi::MpiComm::world().getRank());
    if (mConnectionFutures.at(connectionId).valid())
    {
        // wait for the connection to be created
        mConnectionFutures.at(connectionId).get();
    }
    TLLM_CHECK_WITH_INFO(mConnections.find(connectionId) != mConnections.end(),
        "Connection not found In recvConnect connectionId: %lu , worldRank: %d", connectionId,
        mpi::MpiComm::world().getRank());

    TLLM_CHECK(!mConnections[connectionId]->isFromRequester());

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "recvConnect connectionId: %lu , sendIDData:%lu", connectionId,
        *reinterpret_cast<uint64_t*>(buffer.data()));

    return mConnections[connectionId].get();
}

std::vector<Connection const*> UcxConnectionManager::getConnections(CommState const& state)
{
    std::vector<Connection const*> ret;
    TLLM_CHECK(state.isSocketState());
    for (auto&& scoketState : state.getSocketState())
    {
        std::string address = scoketState.mIp + ":" + std::to_string(scoketState.mPort);
        bool found = false;
        {
            std::scoped_lock lock(mAddressToConnectionIdMutex);
            found = mAddressToConnectionId.find(address) != mAddressToConnectionId.end();
        }
        if (!found)
        {
            addConnection(scoketState.mIp, scoketState.mPort);
        }
        std::scoped_lock lock(mConnectionsMutex, mAddressToConnectionIdMutex);
        TLLM_CHECK(mAddressToConnectionId.find(address) != mAddressToConnectionId.end());
        TLLM_CHECK(mConnections.find(mAddressToConnectionId[address]) != mConnections.end());
        TLLM_CHECK(mConnections[mAddressToConnectionId[address]]->isFromRequester());

        ret.emplace_back(mConnections[mAddressToConnectionId[address]].get());
    }
    return ret;
}

CommState const& UcxConnectionManager::getCommState() const
{
    return mCommState;
}
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> makeUcxConnectionManager()
{
    try
    {
        return UcxConnectionManager::create();
    }
    catch (std::exception const& e)
    {
        std::string error = "Error in makeUcxConnectionManager: " + std::string(e.what());
        TLLM_THROW(error);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace tensorrt_llm::executor::kv_cache
