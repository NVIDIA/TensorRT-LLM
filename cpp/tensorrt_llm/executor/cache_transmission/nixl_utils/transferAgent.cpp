/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cuda.h>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <nixl_types.h>
#include <numeric>
#include <set>
#include <sys/file.h>
#include <sys/stat.h>
#include <thread>
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

[[nodiscard]] MemoryDescs NixlHelper::coalesceMemoryDescs(MemoryDescs const& descs)
{
    auto const& descVec = descs.getDescs();

    // If empty or single element, return as-is
    if (descVec.size() <= 1)
    {
        return descs;
    }

    size_t const numDescs = descVec.size();

    // Create index array and sort by address
    std::vector<size_t> sortedIndices(numDescs);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&descVec](size_t lhs, size_t rhs)
        {
            // Sort by deviceId first, then by address
            if (descVec[lhs].getDeviceId() != descVec[rhs].getDeviceId())
            {
                return descVec[lhs].getDeviceId() < descVec[rhs].getDeviceId();
            }
            return descVec[lhs].getAddr() < descVec[rhs].getAddr();
        });

    std::vector<MemoryDesc> coalesced;
    coalesced.reserve(numDescs);

    // Start with the first entry
    size_t firstIdx = sortedIndices[0];
    uintptr_t currentAddr = descVec[firstIdx].getAddr();
    size_t currentLen = descVec[firstIdx].getLen();
    uint32_t currentDeviceId = descVec[firstIdx].getDeviceId();

    for (size_t idx = 1; idx < numDescs; ++idx)
    {
        size_t sortedIdx = sortedIndices[idx];
        auto const& desc = descVec[sortedIdx];

        // Check if current can be coalesced with previous
        bool isContiguous = (currentAddr + currentLen == desc.getAddr()) && (currentDeviceId == desc.getDeviceId());

        if (isContiguous)
        {
            // Coalesce: extend the current region
            currentLen += desc.getLen();
        }
        else
        {
            // Cannot coalesce: save the current region and start a new one
            coalesced.emplace_back(currentAddr, currentLen, currentDeviceId);

            currentAddr = desc.getAddr();
            currentLen = desc.getLen();
            currentDeviceId = desc.getDeviceId();
        }
    }

    // Add the last region
    coalesced.emplace_back(currentAddr, currentLen, currentDeviceId);

    TLLM_LOG_DEBUG("NixlHelper::coalesceMemoryDescs: coalesced %zu -> %zu entries", descVec.size(), coalesced.size());

    return MemoryDescs{descs.getType(), std::move(coalesced)};
}

[[nodiscard]] std::pair<MemoryDescs, MemoryDescs> NixlHelper::coalesceTransferDescs(
    TransferDescs const& srcDescs, TransferDescs const& dstDescs)
{
    auto const& srcVec = srcDescs.getDescs();
    auto const& dstVec = dstDescs.getDescs();

    // If sizes don't match or empty, return as-is
    if (srcVec.size() != dstVec.size() || srcVec.empty())
    {
        return {srcDescs, dstDescs};
    }

    size_t const numDescs = srcVec.size();

    // Create index array and sort by src address
    // This allows us to find contiguous regions even if the original order is scattered
    std::vector<size_t> sortedIndices(numDescs);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&srcVec](size_t lhs, size_t rhs)
        {
            // Sort by deviceId first, then by address
            if (srcVec[lhs].getDeviceId() != srcVec[rhs].getDeviceId())
            {
                return srcVec[lhs].getDeviceId() < srcVec[rhs].getDeviceId();
            }
            return srcVec[lhs].getAddr() < srcVec[rhs].getAddr();
        });

    std::vector<MemoryDesc> coalescedSrc;
    std::vector<MemoryDesc> coalescedDst;
    coalescedSrc.reserve(numDescs);
    coalescedDst.reserve(numDescs);

    // Start with the first entry (using sorted order)
    size_t firstIdx = sortedIndices[0];
    uintptr_t currentSrcAddr = srcVec[firstIdx].getAddr();
    size_t currentSrcLen = srcVec[firstIdx].getLen();
    uint32_t currentSrcDeviceId = srcVec[firstIdx].getDeviceId();

    uintptr_t currentDstAddr = dstVec[firstIdx].getAddr();
    size_t currentDstLen = dstVec[firstIdx].getLen();
    uint32_t currentDstDeviceId = dstVec[firstIdx].getDeviceId();

    for (size_t idx = 1; idx < numDescs; ++idx)
    {
        size_t sortedIdx = sortedIndices[idx];
        auto const& src = srcVec[sortedIdx];
        auto const& dst = dstVec[sortedIdx];

        // Check if current src and dst can be coalesced with previous
        bool srcContiguous
            = (currentSrcAddr + currentSrcLen == src.getAddr()) && (currentSrcDeviceId == src.getDeviceId());
        bool dstContiguous
            = (currentDstAddr + currentDstLen == dst.getAddr()) && (currentDstDeviceId == dst.getDeviceId());

        if (srcContiguous && dstContiguous)
        {
            // Coalesce: extend the current region
            currentSrcLen += src.getLen();
            currentDstLen += dst.getLen();
        }
        else
        {
            // Cannot coalesce: save the current region and start a new one
            coalescedSrc.emplace_back(currentSrcAddr, currentSrcLen, currentSrcDeviceId);
            coalescedDst.emplace_back(currentDstAddr, currentDstLen, currentDstDeviceId);

            currentSrcAddr = src.getAddr();
            currentSrcLen = src.getLen();
            currentSrcDeviceId = src.getDeviceId();

            currentDstAddr = dst.getAddr();
            currentDstLen = dst.getLen();
            currentDstDeviceId = dst.getDeviceId();
        }
    }

    // Don't forget to add the last region
    coalescedSrc.emplace_back(currentSrcAddr, currentSrcLen, currentSrcDeviceId);
    coalescedDst.emplace_back(currentDstAddr, currentDstLen, currentDstDeviceId);

    TLLM_LOG_DEBUG(
        "NixlHelper::coalesceTransferDescs: coalesced %zu -> %zu transfer entries", srcVec.size(), coalescedSrc.size());

    return {MemoryDescs{srcDescs.getType(), std::move(coalescedSrc)},
        MemoryDescs{dstDescs.getType(), std::move(coalescedDst)}};
}

TransferState NixlTransferStatus::wait(int64_t timeout_ms) const
{
    auto startTime = std::chrono::steady_clock::now();

    while (true)
    {
        auto status = mRawAgent->getXferStatus(mHandle);
        if (status == NIXL_SUCCESS)
        {
            return TransferState::kSUCCESS;
        }
        else if (status != NIXL_IN_PROG)
        {
            return TransferState::kFAILURE;
        }

        // If timeout_ms < 0, wait indefinitely until status is not NIXL_IN_PROG
        if (timeout_ms < 0)
        {
            std::this_thread::yield();
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

        std::this_thread::yield();
    }
}

[[nodiscard]] bool NixlTransferStatus::isCompleted() const
{
    return mRawAgent->getXferStatus(mHandle) == NIXL_SUCCESS;
}

NixlTransferAgent::NixlTransferAgent(BaseAgentConfig const& config)
    : mName{config.mName}
{
    nixl_status_t status;
    if (config.useListenThread)
    {
        FileLock lock("/tmp/trtllm_nixl_port.lock");
        if (!lock.lock())
        {
            TLLM_THROW("Failed to lock /tmp/trtllm_nixl_port.lock");
        }
        auto envPort = common::getEnvNixlPort();
        uint16_t port = envPort > 0 ? getIncrmentPort(envPort) : getAvailablePort();
        uint32_t numWorker = config.backendParams.find("num_workers") != config.backendParams.end()
            ? std::stoi(config.backendParams.at("num_workers"))
            : 1;
        nixlAgentConfig nixlConfig{config.useProgThread, true, port, nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT,
            numWorker, 0, 10000, config.enableTelemetry};
        mAddress = getAvailableIP() + ":" + std::to_string(port);
        mRawAgent = std::make_unique<nixlAgent>(config.mName, std::move(nixlConfig));
    }
    else
    {
        uint32_t numWorker = config.backendParams.find("num_workers") != config.backendParams.end()
            ? std::stoi(config.backendParams.at("num_workers"))
            : 1;
        mAddress.clear();
        nixlAgentConfig nixlConfig{config.useProgThread, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT,
            numWorker, 0, 10000, config.enableTelemetry};
        mRawAgent = std::make_unique<nixlAgent>(config.mName, std::move(nixlConfig));
    }

    std::string nixlBackend = common::getEnvNixlBackend();
    // List of supported backends - extend this list as new backends are added
    static std::set<std::string> const kSUPPORTED_BACKENDS = {"UCX", "LIBFABRIC"};

    if (kSUPPORTED_BACKENDS.find(nixlBackend) == kSUPPORTED_BACKENDS.end())
    {
        TLLM_LOG_WARNING("Unsupported NIXL backend: %s, fallback to UCX", nixlBackend.c_str());
        nixlBackend = "UCX";
    }

    TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent using NIXL backend: %s", nixlBackend.c_str());

    // Get default plugin params first, then override with user-provided backendParams
    // NOTE: getPluginParams overwrites init1, so we must call it BEFORE setting user params
    nixl_b_params_t init1;
    nixl_mem_list_t mems1;
    status = mRawAgent->getPluginParams(nixlBackend.c_str(), mems1, init1);
    TLLM_CHECK(status == NIXL_SUCCESS);

    // Override default params with user-provided backendParams
    for (auto const& [key, value] : config.backendParams)
    {
        init1[key] = value;
        TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent backendParams: %s: %s", key.c_str(), value.c_str());
    }

    status = mRawAgent->createBackend(nixlBackend.c_str(), init1, mRawBackend);
    if (status != NIXL_SUCCESS || !mRawBackend)
    {
        TLLM_THROW("Failed to create NIXL backend: %s", nixlBackend.c_str());
    }
    mExtraParams.backends.push_back(mRawBackend);
    TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent mAddress: %s", mAddress.c_str());
    mDRamSrcBuffer.resize(16);
    mDRamDstBuffer.resize(16);
    MemoryDescs descs{MemoryType::kDRAM, {MemoryDesc{mDRamSrcBuffer}, MemoryDesc{mDRamDstBuffer}}};
    registerMemory(descs);

    // P2P fast-path agent. Constructed after the NIXL agent so that it queries the CUDA
    // device context already set up by NIXL. Disabled via TRTLLM_KV_TRANSFER_P2P_DISABLE.
    if (!common::getEnvKvTransferP2pDisable())
    {
        mP2pAgent = std::make_unique<P2pTransferAgent>();
    }
}

void NixlTransferAgent::registerMemory(RegisterDescs const& descs)
{
    // Split VRAM descriptors at VMM chunk boundaries so each sub-descriptor
    // falls within a single cuMemCreate allocation (required by gdr_copy / cuda_ipc).
    size_t detectedChunkSize = 0;
    auto splitDescs = VmmDescSplitter::splitVmmDescs(descs, detectedChunkSize);

    // Record per-desc VMM chunk info for use in deregisterMemory / submitTransferRequests
    auto detectedRegionMap = VmmDescSplitter::detectVramRegionMap(descs);
    mLocalVramRegionInfo.merge(detectedRegionMap);

    // Coalesce contiguous memory regions to reduce registration overhead (disabled by default)
    // Set TRTLLM_NIXL_ENABLE_COALESCE=1 to enable this optimization
    auto coalescedDescs = common::getEnvNixlEnableCoalesce() ? NixlHelper::coalesceMemoryDescs(splitDescs) : splitDescs;

    nixl_status_t status;
    status = mRawAgent->registerMem(NixlHelper::convertRegDlist(coalescedDescs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    std::string localMD;
    status = mRawAgent->getLocalMD(localMD);
    TLLM_CHECK(status == NIXL_SUCCESS);

    // Export P2P handles for VRAM. This runs its own cuMemGetAddressRange-based scan
    // independent of VmmDescSplitter — the two paths record different information.
    if (descs.getType() == MemoryType::kVRAM && mP2pAgent)
    {
        mP2pAgent->exporter().exportHandles(descs);
    }
}

void NixlTransferAgent::deregisterMemory(RegisterDescs const& descs)
{
    // Split using per-region registry info to match what was registered
    auto splitDescs = VmmDescSplitter::splitDescsWithRegionMap(descs, mLocalVramRegionInfo);

    // Coalesce contiguous memory regions to match what was registered (disabled by default)
    // Set TRTLLM_NIXL_ENABLE_COALESCE=1 to enable this optimization
    auto coalescedDescs = common::getEnvNixlEnableCoalesce() ? NixlHelper::coalesceMemoryDescs(splitDescs) : splitDescs;

    nixl_status_t status;
    status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(coalescedDescs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    // Remove entries from registry
    if (descs.getType() == MemoryType::kVRAM)
    {
        for (auto const& desc : descs.getDescs())
        {
            mLocalVramRegionInfo.erase(desc.getAddr());
        }
        if (mP2pAgent)
        {
            mP2pAgent->exporter().removeHandles(descs);
        }
    }
}

void NixlTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc)
{
    nixl_status_t status;
    std::string remoteName;
    status = mRawAgent->loadRemoteMD(agentDesc.getBackendAgentDesc(), remoteName);
    TLLM_CHECK(status == NIXL_SUCCESS);
    TLLM_CHECK_WITH_INFO(
        name == remoteName, "loadRemoteAgent gets error agent name: %s != %s", name.c_str(), remoteName.c_str());

    // Store remote VMM region info for chunk boundary calculations in
    // VmmDescSplitter::splitTransferDescsWithRegionMaps. Per-agent map because different remote agents may have
    // overlapping virtual addresses.
    auto const& regions = agentDesc.getVramRegions();
    if (!regions.empty())
    {
        auto& remoteMap = mRemoteVramRegionInfo[name];
        for (auto const& r : regions)
        {
            remoteMap[r.baseAddr] = {r.totalLen, r.chunkSize};
        }
    }

    // Import P2P memory from the peer (fabric / POSIX FD / CUDA IPC handles).
    if (mP2pAgent && !agentDesc.getP2pBlob().empty())
    {
        auto info = P2pMemInfo::deserialize(agentDesc.getP2pBlob());
        if (info.has_value() && info->supported)
        {
            mP2pAgent->registry().importAndMap(name, *info);
        }
    }
}

AgentDesc NixlTransferAgent::getLocalAgentDesc()
{
    nixl_blob_t nixlBlob;
    nixl_status_t status = mRawAgent->getLocalMD(nixlBlob);
    TLLM_CHECK(status == NIXL_SUCCESS);

    // Pack local VMM region info so remote agents can compute chunk boundaries.
    std::vector<VramRegionMeta> regions;
    for (auto const& [base, info] : mLocalVramRegionInfo)
    {
        if (info.chunkSize > 0)
        {
            regions.push_back({base, info.totalLen, info.chunkSize});
        }
    }

    // Single locked call inside the exporter — atomic supported-check + serialize.
    // The previous "isSupported() then getLocalInfo().serialize()" pair raced with
    // concurrent registerMemory, which could mutate mLocalInfo between the two reads.
    std::string p2pBlob = mP2pAgent ? mP2pAgent->serializeLocalInfoIfSupported() : std::string{};

    return AgentDesc{nixlBlob, std::move(regions), std::move(p2pBlob)};
}

void NixlTransferAgent::invalidateRemoteAgent(std::string const& name)
{
    // Clean up remote VMM region info before invalidating the remote agent.
    mRemoteVramRegionInfo.erase(name);
    if (mP2pAgent)
    {
        mP2pAgent->registry().cleanup(name);
    }
    mRawAgent->invalidateRemoteMD(name);
}

// Build and post a NIXL xfer request. Shared by the pure-NIXL path (all segments or no
// P2P mapping available) and the mixed path (only the segments that failed P2P translate
// are forwarded here). Ownership of the returned handle moves into NixlTransferStatus.
[[nodiscard]] std::unique_ptr<TransferStatus> NixlTransferAgent::submitNixlTransferInternal(TransferOp op,
    MemoryDescs const& srcDescs, MemoryDescs const& dstDescs, std::string const& remoteName,
    std::optional<SyncMessage> const& syncMessage)
{
    nixl_status_t status;
    nixlXferReqH* handle;

    // Use a local copy of mExtraParams so two concurrent submitters don't race on hasNotif/notifMsg.
    nixl_opt_args_t localParams = mExtraParams;
    if (syncMessage.has_value())
    {
        localParams.hasNotif = true;
        localParams.notifMsg = syncMessage.value();
    }
    else
    {
        localParams.hasNotif = false;
    }

    static VramRegionMap const kEmptyMap;
    auto remoteIt = mRemoteVramRegionInfo.find(remoteName);
    auto const& remoteRegionMap = (remoteIt != mRemoteVramRegionInfo.end()) ? remoteIt->second : kEmptyMap;

    auto [splitSrc, splitDst]
        = VmmDescSplitter::splitTransferDescsWithRegionMaps(srcDescs, dstDescs, mLocalVramRegionInfo, remoteRegionMap);

    if (common::getEnvNixlEnableCoalesce())
    {
        NVTX3_SCOPED_RANGE(coalesceTransferDescs_CreateXferReq);
        auto [coalescedSrc, coalescedDst] = NixlHelper::coalesceTransferDescs(splitSrc, splitDst);
        status = mRawAgent->createXferReq(NixlHelper::convert(op), NixlHelper::convertXferDist(coalescedSrc),
            NixlHelper::convertXferDist(coalescedDst), remoteName, handle, &localParams);
    }
    else
    {
        status = mRawAgent->createXferReq(NixlHelper::convert(op), NixlHelper::convertXferDist(splitSrc),
            NixlHelper::convertXferDist(splitDst), remoteName, handle, &localParams);
    }

    TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS,
        " rank: %d createXferReq failed with status: %s selfname: %s remoteAgent name: %s",
        mpi::MpiComm::world().getRank(), nixlEnumStrings::statusStr(status).c_str(), mName.c_str(), remoteName.c_str());
    {
        NVTX3_SCOPED_RANGE(postXferReq);
        status = mRawAgent->postXferReq(handle, &localParams);
    }
    return std::make_unique<NixlTransferStatus>(mRawAgent.get(), handle);
}

[[nodiscard]] std::unique_ptr<TransferStatus> NixlTransferAgent::submitTransferRequests(TransferRequest const& request)
{
    NVTX3_SCOPED_RANGE(NixlTransferAgent_submitTransferRequests);

    // ---- P2P eligibility ----
    // P2P is skipped entirely when:
    // - No P2P agent configured
    // - Either endpoint is not VRAM
    // - Request carries a sync message (NIXL's notification channel is required)
    // - No remote mapping exists for this peer (import never ran or was cleaned up)
    // - Mixed mode is disabled via env and the mapping is partial (old all-or-nothing behavior)
    //
    // Cache the shared_ptr once so translate() in the per-segment loop is lock-free.
    std::shared_ptr<RemoteP2pMapping const> remoteMapping
        = (mP2pAgent && !request.getSyncMessage().has_value() && request.getSrcDescs().getType() == MemoryType::kVRAM
              && request.getDstDescs().getType() == MemoryType::kVRAM)
        ? mP2pAgent->registry().get(request.getRemoteName())
        : nullptr;

    if (remoteMapping == nullptr)
    {
        // ---- Pure NIXL path ----
        mPureNixlCount.fetch_add(1, std::memory_order_relaxed);
        return submitNixlTransferInternal(request.getOp(), request.getSrcDescs(), request.getDstDescs(),
            request.getRemoteName(), request.getSyncMessage());
    }

    auto const& srcDescs = request.getSrcDescs().getDescs();
    auto const& dstDescs = request.getDstDescs().getDescs();
    size_t const numSegments = srcDescs.size();
    TLLM_CHECK_WITH_INFO(numSegments == dstDescs.size(), "Transfer: srcDescs.size(%zu) != dstDescs.size(%zu)",
        numSegments, dstDescs.size());

    bool const isWrite = (request.getOp() == TransferOp::kWRITE);

    // Bucket each segment into either P2P (translate returned a local mapped ptr) or NIXL
    // (translate returned nullptr — unmapped pool). A single for-loop visits every segment;
    // there is no early-break like the old all-or-nothing logic.
    std::vector<void*> p2pSrcPtrs;
    std::vector<void*> p2pDstPtrs;
    std::vector<size_t> p2pSizes;
    p2pSrcPtrs.reserve(numSegments);
    p2pDstPtrs.reserve(numSegments);
    p2pSizes.reserve(numSegments);

    std::vector<MemoryDesc> nixlSrcVec;
    std::vector<MemoryDesc> nixlDstVec;
    nixlSrcVec.reserve(numSegments);
    nixlDstVec.reserve(numSegments);

    size_t p2pBytes = 0;

    for (size_t i = 0; i < numSegments; ++i)
    {
        size_t const segSize = srcDescs[i].getLen();
        // NIXL convention: dstDescs carry REMOTE addresses on the peer.
        void* mappedRemotePtr = P2pRemoteMappingRegistry::translate(*remoteMapping, dstDescs[i].getAddr(), segSize);
        if (mappedRemotePtr == nullptr)
        {
            nixlSrcVec.push_back(srcDescs[i]);
            nixlDstVec.push_back(dstDescs[i]);
            continue;
        }
        void* localPtr = reinterpret_cast<void*>(srcDescs[i].getAddr());
        p2pBytes += segSize;
        p2pSrcPtrs.push_back(isWrite ? localPtr : mappedRemotePtr);
        p2pDstPtrs.push_back(isWrite ? mappedRemotePtr : localPtr);
        p2pSizes.push_back(segSize);
    }

    bool const havePartialFallback = !nixlSrcVec.empty();
    bool const mixedDisabled = common::getEnvKvTransferP2pMixedDisable();
    if (havePartialFallback && mixedDisabled)
    {
        // Safety valve: user has asked to preserve the pre-mixed all-or-nothing behavior.
        // If any segment missed, push the ENTIRE request through NIXL.
        TLLM_LOG_WARNING("P2pTransfer: mapping incomplete for '%s' and mixed-mode disabled -> full NIXL fallback",
            request.getRemoteName().c_str());
        mPureNixlCount.fetch_add(1, std::memory_order_relaxed);
        return submitNixlTransferInternal(request.getOp(), request.getSrcDescs(), request.getDstDescs(),
            request.getRemoteName(), request.getSyncMessage());
    }

    // Helper to submit the P2P half via the appropriate path (cub vs memcpyBatch).
    auto submitP2pHalf = [&]() -> std::unique_ptr<TransferStatus>
    {
        if (p2pSrcPtrs.empty())
        {
            return nullptr;
        }
        size_t const avgSegmentSize = p2pBytes / p2pSrcPtrs.size();
        size_t const thresholdBytes = common::getEnvKvTransferP2pBatchThresholdKB() * 1024;
        auto& ctx = mP2pAgent->contextForCurrentThread();
        if (avgSegmentSize < thresholdBytes)
        {
            mCubSubmitCount.fetch_add(1, std::memory_order_relaxed);
            TLLM_LOG_DEBUG("P2pTransfer: cub path %zu/%zu segs avgSize=%zuB total=%zuB to %s (op=%s)",
                p2pSrcPtrs.size(), numSegments, avgSegmentSize, p2pBytes, request.getRemoteName().c_str(),
                isWrite ? "WRITE" : "READ");
            return ctx.submitWithCubBatched(p2pSrcPtrs, p2pDstPtrs, p2pSizes);
        }
        mMemcpyBatchSubmitCount.fetch_add(1, std::memory_order_relaxed);
        TLLM_LOG_DEBUG("P2pTransfer: memcpyBatch path %zu/%zu segs avgSize=%zuB total=%zuB to %s (op=%s)",
            p2pSrcPtrs.size(), numSegments, avgSegmentSize, p2pBytes, request.getRemoteName().c_str(),
            isWrite ? "WRITE" : "READ");
        return ctx.submitWithMemcpyBatch(p2pSrcPtrs, p2pDstPtrs, p2pSizes);
    };

    // ---- Case A: every segment mapped -> P2P only (same as old fast path). ----
    if (!havePartialFallback)
    {
        mPureP2pCount.fetch_add(1, std::memory_order_relaxed);
        return submitP2pHalf();
    }

    // ---- Case B: no segment mapped -> NIXL only.
    // translate() returned nullptr for every segment, which is unusual but possible
    // (e.g. remote agent was just cleaned up mid-request, or all pools happen to be
    // on the unmapped side). Equivalent to the pre-mixed fallback branch.
    if (p2pSrcPtrs.empty())
    {
        mPureNixlCount.fetch_add(1, std::memory_order_relaxed);
        return submitNixlTransferInternal(request.getOp(), request.getSrcDescs(), request.getDstDescs(),
            request.getRemoteName(), request.getSyncMessage());
    }

    // ---- Case C: mixed -> submit both halves, return a composite status. ----
    // NIXL side uses only the unmapped subset. Submit P2P FIRST so its CUDA stream is
    // launched ASAP (any host-side overhead of createXferReq runs while the GPU works).
    mMixedCount.fetch_add(1, std::memory_order_relaxed);
    TLLM_LOG_DEBUG("P2pTransfer: mixed path for '%s' — P2P %zu segs, NIXL %zu segs", request.getRemoteName().c_str(),
        p2pSrcPtrs.size(), nixlSrcVec.size());

    auto p2pStatus = submitP2pHalf();

    MemoryDescs nixlSrcDescs{request.getSrcDescs().getType(), std::move(nixlSrcVec)};
    MemoryDescs nixlDstDescs{request.getDstDescs().getType(), std::move(nixlDstVec)};
    auto nixlStatus = submitNixlTransferInternal(
        request.getOp(), nixlSrcDescs, nixlDstDescs, request.getRemoteName(), request.getSyncMessage());

    return std::make_unique<MixedTransferStatus>(std::move(p2pStatus), std::move(nixlStatus));
}

PathCounterSnapshot NixlTransferAgent::getPathCounters() const noexcept
{
    PathCounterSnapshot snap{};
    snap.pureP2p = mPureP2pCount.load(std::memory_order_relaxed);
    snap.mixed = mMixedCount.load(std::memory_order_relaxed);
    snap.pureNixl = mPureNixlCount.load(std::memory_order_relaxed);
    snap.cubSubmit = mCubSubmitCount.load(std::memory_order_relaxed);
    snap.memcpyBatchSubmit = mMemcpyBatchSubmitCount.load(std::memory_order_relaxed);
    if (mP2pAgent)
    {
        auto p2pSnap = mP2pAgent->getCountersSnapshot();
        snap.memcpyBatchSingleThread = p2pSnap.memcpyBatchSingleThread;
        snap.memcpyBatchMultiThread = p2pSnap.memcpyBatchMultiThread;
    }
    else
    {
        snap.memcpyBatchSingleThread = 0;
        snap.memcpyBatchMultiThread = 0;
    }
    return snap;
}

void NixlTransferAgent::notifySyncMessage(std::string const& name, SyncMessage const& syncMessage)
{

    auto status = mRawAgent->genNotif(name, syncMessage);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "genNotif failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
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
    TransferState transferState = status->wait();
    TLLM_CHECK_WITH_INFO(transferState == TransferState::kSUCCESS, "submitLoopbackRequests failed");

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
