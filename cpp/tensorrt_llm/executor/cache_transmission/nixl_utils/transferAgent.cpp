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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <nixl_types.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

class FileLock
{
private:
    int fd_;
    std::string lockFile_;
    bool locked_;

public:
    explicit FileLock(std::string const& lockFile)
        : fd_(-1)
        , lockFile_(lockFile)
        , locked_(false)
    {
    }

    ~FileLock()
    {
        unlock();
    }

    bool lock()
    {
        if (locked_)
            return true;

        size_t pos = lockFile_.find_last_of('/');
        if (pos != std::string::npos)
        {
            std::string dir = lockFile_.substr(0, pos);
            mkdir(dir.c_str(), 0755);
        }

        fd_ = open(lockFile_.c_str(), O_CREAT | O_WRONLY, 0644);
        if (fd_ == -1)
        {
            TLLM_LOG_ERROR("Failed to open lock file: %s", lockFile_.c_str());
            return false;
        }

        if (flock(fd_, LOCK_EX) == -1)
        {
            TLLM_LOG_ERROR("Failed to acquire file lock: %s", lockFile_.c_str());
            close(fd_);
            fd_ = -1;
            return false;
        }

        locked_ = true;
        return true;
    }

    void unlock()
    {
        if (locked_ && fd_ != -1)
        {
            flock(fd_, LOCK_UN);
            close(fd_);
            fd_ = -1;
            locked_ = false;
        }
    }
};

static std::string getAvailableIP()
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

        std::string nixlInterface = common::getEnvNixlInterface();
        if (!nixlInterface.empty() && strcmp(ifa->ifa_name, nixlInterface.c_str()) != 0)
        {
            continue;
        }

        // Skip the loopback interface
        if (nixlInterface.empty() && (strncmp(ifa->ifa_name, "docker", 6) == 0 || strcmp(ifa->ifa_name, "lo") == 0))
        {
            continue;
        }

        // Check if the address family is AF_INET (IPv4)
        // TODO: USER CAN SPECIFY THE IP ADDRESS
        if (ifa->ifa_addr->sa_family == AF_INET)
        {
            addr_ptr = &((struct sockaddr_in*) ifa->ifa_addr)->sin_addr;
            char address_buffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, addr_ptr, address_buffer, sizeof(address_buffer));

            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " ***** NIXL    Interface: %s IP Address: %s",
                ifa->ifa_name, address_buffer);
            ip = address_buffer;
            break;
        }
    }
    if (ifa == nullptr)
    {
        TLLM_LOG_ERROR(mpi::MpiComm::world().getRank(),
            "UCX   No valid IP address found please set correct NIXL interface with env variable TRTLLM_UCX_INTERFACE");
    }

    freeifaddrs(ifaddr);
    return ip;
}

uint16_t getAvailablePort(std::string const& ip = "0.0.0.0")
{
    struct addrinfo hints
    {
    };

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo* res;
    int ret = getaddrinfo(ip.c_str(), "0", &hints, &res);
    TLLM_CHECK(ret == 0);

    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    TLLM_CHECK(sockfd != -1);

    ret = bind(sockfd, res->ai_addr, res->ai_addrlen);
    TLLM_CHECK(ret == 0);

    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    ret = getsockname(sockfd, (struct sockaddr*) &addr, &addrlen);
    TLLM_CHECK(ret == 0);

    uint16_t port = ntohs(addr.sin_port);
    close(sockfd);
    freeaddrinfo(res);

    return port;
}

uint16_t getIncrmentPort(uint16_t basePort)
{
    static uint16_t times = 0;
    return basePort + mpi::MpiComm::world().getRank() + (times++) * mpi::MpiComm::world().getSize();
    // just for test
}

[[nodiscard]] nixl_mem_t NixlHelper::convert(MemoryType type)
{
    switch (type)
    {
    case MemoryType::kDRAM: return DRAM_SEG;
    case MemoryType::kVRAM: return VRAM_SEG;
    case MemoryType::kBLK: return BLK_SEG;
    case MemoryType::kOBJ: return OBJ_SEG;
    case MemoryType::kFILE: return FILE_SEG;
    default: TLLM_THROW("Unknown MemoryType value");
    }
}

[[nodiscard]] nixlBasicDesc NixlHelper::convert(MemoryDesc const& desc)
{
    return nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()};
}

[[nodiscard]] nixl_reg_dlist_t NixlHelper::convertRegDlist(RegisterDescs const& descs)
{
    nixl_reg_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBlobDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

[[nodiscard]] nixl_reg_dlist_t NixlHelper::convertRegDlist(FileDescs const& descs)
{
    nixl_reg_dlist_t list(FILE_SEG);
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBlobDesc{0, desc.getLen(), desc.getFd()});
    }
    return list;
}

[[nodiscard]] nixl_xfer_op_t NixlHelper::convert(TransferOp const& op)
{
    switch (op)
    {
    case TransferOp::kREAD: return NIXL_READ;
    case TransferOp::kWRITE: return NIXL_WRITE;
    default: TLLM_THROW("Unknown TransferOp value");
    }
}

[[nodiscard]] nixl_xfer_dlist_t NixlHelper::convertXferDist(TransferDescs const& descs)
{
    nixl_xfer_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

[[nodiscard]] nixl_xfer_dlist_t NixlHelper::convertXferDist(FileDescs const& descs)
{
    nixl_xfer_dlist_t list{FILE_SEG};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBasicDesc{0, desc.getLen(), desc.getFd()});
    }
    return list;
}

void NixlHelper::posixGpuToFileFallback(MemoryDescs const& memoryDescs, FileDescs const& fileDescs)
{
    auto const& memVec = memoryDescs.getDescs();
    auto const& fileVec = fileDescs.getDescs();
    std::size_t i;

    for (i = 0; i < std::min(memVec.size(), fileVec.size()); i++)
    {
        auto& memDesc = memVec[i];
        auto& fileDesc = fileVec[i];

        ssize_t numBytes = static_cast<ssize_t>(memDesc.getLen());
        std::vector<uint8_t> hostBuffer(numBytes);

        cudaError_t cpyErr = cudaMemcpy(
            hostBuffer.data(), reinterpret_cast<void*>(memDesc.getAddr()), numBytes, cudaMemcpyDeviceToHost);
        TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to host failed, error=%d", cpyErr);

        ssize_t written = ::write(fileDesc.getFd(), hostBuffer.data(), numBytes);
        TLLM_CHECK_WITH_INFO(written >= 0, "POSIX write error=%zd", written);
    }
}

void NixlHelper::posixFileToGpuFallback(MemoryDescs const& memoryDescs, FileDescs const& fileDescs)
{
    auto const& memVec = memoryDescs.getDescs();
    auto const& fileVec = fileDescs.getDescs();
    std::size_t i;

    for (i = 0; i < std::min(memVec.size(), fileVec.size()); i++)
    {
        auto& memDesc = memVec[i];
        auto& fileDesc = fileVec[i];

        ssize_t numBytes = static_cast<ssize_t>(memDesc.getLen());
        std::vector<uint8_t> hostBuffer(numBytes);

        ssize_t bytesRead = ::read(fileDesc.getFd(), hostBuffer.data(), numBytes);
        TLLM_CHECK_WITH_INFO(bytesRead == numBytes, "POSIX read error=%zd", bytesRead);

        cudaError_t cpyErr = cudaMemcpy(
            reinterpret_cast<void*>(memDesc.getAddr()), hostBuffer.data(), numBytes, cudaMemcpyHostToDevice);
        TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to device failed, error=%d", cpyErr);
    }
}

NixlTransferStatus::NixlTransferStatus(nixlAgent* agent, nixlXferReqH* handle)
    : mRawAgent{agent}
    , mHandle{handle}
{
    TLLM_CHECK(mRawAgent);
    TLLM_CHECK(mHandle);
}

void NixlTransferStatus::wait() const
{
    while (!isCompleted())
        ;
}

[[nodiscard]] bool NixlTransferStatus::isCompleted() const
{
    return mRawAgent->getXferStatus(mHandle) == NIXL_SUCCESS;
}

NixlTransferAgent::NixlTransferAgent(BaseAgentConfig const& config)
    : mName{config.mName}
{
    nixl_status_t status;
    {
        FileLock lock("/tmp/trtllm_nixl_port.lock");
        if (!lock.lock())
        {
            TLLM_THROW("Failed to lock /tmp/trtllm_nixl_port.lock");
        }
        auto envPort = common::getEnvNixlPort();
        uint16_t port = envPort > 0 ? getIncrmentPort(envPort) : getAvailablePort();
        nixlAgentConfig nixlConfig{config.useProgThread, true, port};
        mAddress = getAvailableIP() + ":" + std::to_string(port);
        mRawAgent = std::make_unique<nixlAgent>(config.mName, std::move(nixlConfig));
    }

    nixl_b_params_t init1;
    nixl_mem_list_t mems1;
    status = mRawAgent->getPluginParams("UCX", mems1, init1);
    TLLM_CHECK(status == NIXL_SUCCESS);

    status = mRawAgent->createBackend("UCX", init1, mRawBackend);
    if (status != NIXL_SUCCESS || !mRawBackend)
    {
        TLLM_THROW("Failed to create NIXL backend");
    }
    mExtraParams.backends.push_back(mRawBackend);
    TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent mAddress: %s", mAddress.c_str());
    mDRamSrcBuffer.resize(16);
    mDRamDstBuffer.resize(16);
    MemoryDescs descs{MemoryType::kDRAM, {MemoryDesc{mDRamSrcBuffer}, MemoryDesc{mDRamDstBuffer}}};
    registerMemory(descs);
}

void NixlTransferAgent::registerMemory(RegisterDescs const& descs)
{
    nixl_status_t status;
    status = mRawAgent->registerMem(NixlHelper::convertRegDlist(descs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    std::string localMD;
    status = mRawAgent->getLocalMD(localMD);
    TLLM_CHECK(status == NIXL_SUCCESS);
}

void NixlTransferAgent::deregisterMemory(RegisterDescs const& descs)
{
    nixl_status_t status;
    status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(descs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);
}

void NixlTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc)
{
    nixl_status_t status;
    std::string remoteName;
    status = mRawAgent->loadRemoteMD(agentDesc.getBackendAgentDesc(), remoteName);
    TLLM_CHECK(status == NIXL_SUCCESS);
    TLLM_CHECK_WITH_INFO(
        name == remoteName, "loadRemoteAgent gets error agent name: %s != %s", name.c_str(), remoteName.c_str());
}

AgentDesc NixlTransferAgent::getLocalAgentDesc()
{
    nixl_blob_t desc;
    nixl_status_t status = mRawAgent->getLocalMD(desc);
    TLLM_CHECK(status == NIXL_SUCCESS);
    return AgentDesc{desc};
}

void NixlTransferAgent::invalidateRemoteAgent(std::string const& name)
{
    mRawAgent->invalidateRemoteMD(name);
}

[[nodiscard]] std::unique_ptr<TransferStatus> NixlTransferAgent::submitTransferRequests(TransferRequest const& request)
{
    nixl_status_t status;
    nixlXferReqH* handle;

    if (request.getSyncMessage().has_value())
    {
        mExtraParams.hasNotif = true;

        mExtraParams.notifMsg = request.getSyncMessage().value();
    }
    else
    {
        mExtraParams.hasNotif = false;
    }
    // Need to do this in a loop with NIXL_ERR_NOT_FOUND
    // UCX AM with desc list is faster than listener thread can recv/load MD with sockets
    // Will be deprecated with ETCD or callbacks

    // do
    // {
    status = mRawAgent->createXferReq(NixlHelper::convert(request.getOp()),
        NixlHelper::convertXferDist(request.getSrcDescs()), NixlHelper::convertXferDist(request.getDstDescs()),
        request.getRemoteName(), handle, &mExtraParams);
    // } while (status == NIXL_ERR_NOT_FOUND);

    TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS,
        " rank: %d createXferReq failed with status: %s selfname: %s remoteAgent name: %s",
        mpi::MpiComm::world().getRank(), nixlEnumStrings::statusStr(status).c_str(), mName.c_str(),
        request.getRemoteName().c_str());

    status = mRawAgent->postXferReq(handle, &mExtraParams);
    return std::make_unique<NixlTransferStatus>(mRawAgent.get(), handle);
}

void NixlTransferAgent::notifySyncMessage(std::string const& name, SyncMessage const& syncMessage)
{
    if (name == mName)
    {
        // FIXME: nixl does not support gen notif to itself ,but support local transfer. we use local transfer to notify
        // itself
        MemoryDescs descs{MemoryType::kDRAM, {MemoryDesc{mDRamSrcBuffer}, MemoryDesc{mDRamDstBuffer}}};
        TransferRequest request{TransferOp::kWRITE, descs, descs, name, syncMessage};
        auto request_status = submitTransferRequests(request);
        request_status->wait();
    }
    else
    {
        auto status = mRawAgent->genNotif(name, syncMessage);
        TLLM_CHECK_WITH_INFO(
            status == NIXL_SUCCESS, "genNotif failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
    }
}

[[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> NixlTransferAgent::getNotifiedSyncMessages()
{

    nixl_notifs_t notifs;
    auto status = mRawAgent->getNotifs(notifs);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "getNotifs failed with status: %s", nixlEnumStrings::statusStr(status).c_str());

    return notifs;
}

ConnectionInfoType NixlTransferAgent::getLocalConnectionInfo()
{
    return mAddress;
}

void NixlTransferAgent::loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo)
{
    std::string ip = connectionInfo.substr(0, connectionInfo.find(":"));
    std::string port = connectionInfo.substr(connectionInfo.find(":") + 1);
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "NixlTransferAgent::loadRemoteAgent loadRemoteAgent to %s remoteagent name: %s", connectionInfo.c_str(),
        name.c_str());
    TLLM_CHECK_WITH_INFO(!ip.empty() && !port.empty(), "loadRemoteAgent get empty ip or port, connectionInfo: %s",
        connectionInfo.c_str());
    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = ip;
    md_extra_params.port = std::stoi(port);
    auto status = mRawAgent->fetchRemoteMD(name, &md_extra_params);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "fetchRemoteMD failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
    // status = mRawAgent->sendLocalMD(&md_extra_params);
    // TLLM_CHECK_WITH_INFO(
    //     status == NIXL_SUCCESS, "sendLocalMD failed with status: %s", nixlEnumStrings::statusStr(status).c_str());

    status = NIXL_ERR_NOT_FOUND;
    nixl_xfer_dlist_t descs{DRAM_SEG};
    while (status == NIXL_ERR_NOT_FOUND)
    {
        status = mRawAgent->checkRemoteMD(name, descs);
        TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS || status == NIXL_ERR_NOT_FOUND,
            "checkRemoteMD failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
        if (status == NIXL_ERR_NOT_FOUND)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "NixlTransferAgent::loadRemoteAgent loadRemoteAgent to %s remoteagent name: %s success status: %s",
        connectionInfo.c_str(), name.c_str(), nixlEnumStrings::statusStr(status).c_str());
}

bool NixlTransferAgent::checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs)
{
    auto status = mRawAgent->checkRemoteMD(name, NixlHelper::convertXferDist(memoryDescs));
    TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS || status == NIXL_ERR_NOT_FOUND, "checkRemoteMD failed with status: %s",
        nixlEnumStrings::statusStr(status).c_str());
    return status == NIXL_SUCCESS;
}

NixlTransferAgent::~NixlTransferAgent()
{
    TLLM_LOG_DEBUG("NixlTransferAgent::~NixlTransferAgent");
}

NixlLoopbackAgent::NixlLoopbackAgent(BaseAgentConfig const& config)
    : mName{config.mName}
{
    nixlAgentConfig nixlConfig{config.useProgThread};
    nixlBackendH* backend;
    nixl_status_t status;
    nixl_b_params_t init;

    mRawAgent = std::make_unique<nixlAgent>(config.mName, std::move(nixlConfig));
    init["batch_pool_size"] = std::to_string(8);
    init["batch_limit"] = std::to_string(128);
    init["max_request_size"] = std::to_string(16 * 1024 * 1024);

    if (config.multiThread)
    {
        status = mRawAgent->createBackend("GDS_MT", init, backend);
        if (status != NIXL_SUCCESS || !backend)
            TLLM_THROW("Failed to create NIXL GDS_MT backend, status = %d", status);
    }
    else
    {
        status = mRawAgent->createBackend("GDS", init, backend);
        if (status != NIXL_SUCCESS || !backend)
            TLLM_THROW("Failed to create NIXL GDS backend, status = %d", status);
    }
}

int NixlLoopbackAgent::registerMemory(MemoryDescs const& descs)
{
    nixl_status_t status = mRawAgent->registerMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

int NixlLoopbackAgent::deregisterMemory(MemoryDescs const& descs)
{
    nixl_status_t status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

int NixlLoopbackAgent::registerFiles(FileDescs const& descs)
{
    nixl_status_t status = mRawAgent->registerMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

int NixlLoopbackAgent::deregisterFiles(FileDescs const& descs)
{
    nixl_status_t status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

std::unique_ptr<TransferStatus> NixlLoopbackAgent::submitLoopbackRequests(
    MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload)
{
    nixl_xfer_dlist_t vram_seg = NixlHelper::convertXferDist(memoryDescs);
    nixl_xfer_dlist_t file_seg = NixlHelper::convertXferDist(fileDescs);
    nixl_xfer_dlist_t& src = isOffload ? vram_seg : file_seg;
    nixl_xfer_dlist_t& dst = isOffload ? file_seg : vram_seg;
    nixl_xfer_op_t op = isOffload ? NIXL_WRITE : NIXL_READ;
    nixlXferReqH* handle = nullptr;

    nixl_status_t status = mRawAgent->createXferReq(op, src, dst, mName, handle);
    TLLM_CHECK(status == NIXL_SUCCESS && handle);
    status = mRawAgent->postXferReq(handle);
    TLLM_CHECK(status == NIXL_IN_PROG);

    return std::make_unique<NixlTransferStatus>(mRawAgent.get(), handle);
}

void NixlLoopbackAgent::executeLoopbackRequest(
    MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload)
{
    bool fallback = false;
    int ret;

    ret = this->registerFiles(fileDescs);
    if (ret < 0)
    { // register can fail if no GDS support
        TLLM_LOG_DEBUG("NIXL GDS register files failed, using POSIX fallback");
        fallback = true;
    }
    else
    {
        ret = this->registerMemory(memoryDescs);
        if (ret < 0)
        { // register can fail if no GDS support
            TLLM_LOG_DEBUG("NIXL GDS register memory failed, using POSIX fallback");
            this->deregisterFiles(fileDescs);
            fallback = true;
        }
    }

    if (fallback)
    {
        if (isOffload)
        {
            NixlHelper::posixGpuToFileFallback(memoryDescs, fileDescs);
        }
        else
        {
            NixlHelper::posixFileToGpuFallback(memoryDescs, fileDescs);
        }

        return;
    }

    std::unique_ptr<TransferStatus> status = this->submitLoopbackRequests(memoryDescs, fileDescs, isOffload);
    TLLM_CHECK_WITH_INFO(status != nullptr, "submitLoopbackRequests failed");
    status->wait();

    this->deregisterMemory(memoryDescs);
    this->deregisterFiles(fileDescs);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    std::unique_ptr<BaseTransferAgent> createNixlTransferAgent(BaseAgentConfig const* config)
    {
        TLLM_CHECK(config);
        return std::make_unique<NixlTransferAgent>(*config);
    }
}

extern "C"
{
    std::shared_ptr<BaseLoopbackAgent> createNixlLoopbackAgent(BaseAgentConfig const* config)
    {
        TLLM_CHECK(config);
        return std::make_shared<NixlLoopbackAgent>(*config);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
