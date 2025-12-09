/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <ifaddrs.h>
#include <netdb.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <variant>
#include <vector>

namespace tensorrt_llm::executor
{

class Serialization;

namespace kv_cache
{

// Describe the data structure for cache layout, which can be used to infer
// cache layouts and location associations between different processes, in order
// to determine suitable senders and receivers.
class CacheState final
{
public:
    struct ModelConfig;

    enum class AttentionType : std::uint8_t
    {
        kDEFAULT = 0,
        kMLA = 1,
    };

    CacheState(ModelConfig modelConfig, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        AttentionType attentionType = AttentionType::kDEFAULT, int kvFactor = 2, bool enableBlockReuse = false,
        bool hasIndexerKCache = false, SizeType32 indexerDimPerHead = 0, SizeType32 indexerKCacheQuantBlockSize = 128)
        : mModelConfig(std::move(modelConfig))
        , mParallelConfig{worldConfig.getTensorParallelism(), worldConfig.getPipelineParallelism(),
              worldConfig.getContextParallelism(), worldConfig.enableAttentionDP(), worldConfig.getTensorParallelRank(),
              worldConfig.getTensorParallelism(), attentionLayerNumPerPP}
        , mDataType{dataType}
        , mAttentionConfig(attentionType, kvFactor)
    {
        mEnableBlockReuse = enableBlockReuse;
        mHasIndexerKCache = hasIndexerKCache;
        mIndexerDimPerHead = indexerDimPerHead;
        mIndexerKCacheQuantBlockSize = indexerKCacheQuantBlockSize;
    }

    CacheState(std::vector<SizeType32> nbKvHeadPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, SizeType32 contextParallelism,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        AttentionType attentionType = AttentionType::kDEFAULT, int kvFactor = 2, bool enableAttentionDP = false,
        int DPrank = 0, int DPsize = 0, bool enableBlockReuse = false, bool hasIndexerKCache = false,
        SizeType32 indexerDimPerHead = 0, SizeType32 indexerKCacheQuantBlockSize = 128)
        : mModelConfig{std::move(nbKvHeadPerLayer), sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism, contextParallelism, enableAttentionDP, DPrank, DPsize,
              attentionLayerNumPerPP}
        , mDataType{dataType}
        , mAttentionConfig(attentionType, kvFactor)
    {
        mEnableBlockReuse = enableBlockReuse;
        mHasIndexerKCache = hasIndexerKCache;
        mIndexerDimPerHead = indexerDimPerHead;
        mIndexerKCacheQuantBlockSize = indexerKCacheQuantBlockSize;
    }

    CacheState(SizeType32 nbAttentionLayers, SizeType32 nbKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, SizeType32 contextParallelism,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        AttentionType attentionType = AttentionType::kDEFAULT, int kvFactor = 2, bool enableAttentionDP = false,
        int DPrank = 0, int DPsize = 0, bool enableBlockReuse = false, bool hasIndexerKCache = false,
        SizeType32 indexerDimPerHead = 0, SizeType32 indexerKCacheQuantBlockSize = 128)
        : mModelConfig{std::vector(nbAttentionLayers, nbKvHeads), sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism, contextParallelism, enableAttentionDP, DPrank, DPsize,
              attentionLayerNumPerPP}
        , mDataType{dataType}
        , mAttentionConfig(attentionType, kvFactor)
    {
        mEnableBlockReuse = enableBlockReuse;
        mHasIndexerKCache = hasIndexerKCache;
        mIndexerDimPerHead = indexerDimPerHead;
        mIndexerKCacheQuantBlockSize = indexerKCacheQuantBlockSize;
    }

    [[nodiscard]] bool operator==(kv_cache::CacheState const& other) const noexcept
    {
        return mModelConfig == other.mModelConfig && mParallelConfig == other.mParallelConfig
            && mAttentionConfig == other.mAttentionConfig && mDataType == other.mDataType;
    }

    struct ModelConfig
    {
        std::vector<SizeType32> mNbKvHeadsPerLayer;
        SizeType32 mSizePerHead;
        SizeType32 mTokensPerBlock;

        [[nodiscard]] bool operator==(ModelConfig const& other) const noexcept
        {
            return mNbKvHeadsPerLayer == other.mNbKvHeadsPerLayer && mSizePerHead == other.mSizePerHead
                && mTokensPerBlock == other.mTokensPerBlock;
        }
    };

    struct ParallelConfig
    {
        SizeType32 mTensorParallelism;
        SizeType32 mPipelineParallelism;
        SizeType32 mContextParallelism;
        bool mEnableAttentionDP;
        SizeType32 mDPrank;
        SizeType32 mDPsize;
        // number of attention layers per pipeline parallelism rank, the size of the vector is equal to the pipeline
        // parallelism size.
        std::vector<SizeType32> mAttentionLayerNumPerPP;

        [[nodiscard]] bool operator==(ParallelConfig const& other) const noexcept
        {
            return mTensorParallelism == other.mTensorParallelism && mPipelineParallelism == other.mPipelineParallelism
                && mContextParallelism == other.mContextParallelism && mEnableAttentionDP == other.mEnableAttentionDP
                && mDPrank == other.mDPrank && mDPsize == other.mDPsize
                && mAttentionLayerNumPerPP == other.mAttentionLayerNumPerPP;
        }
    };

    struct AttentionConfig
    {

        AttentionConfig(AttentionType attentionType, int kvFactor)
            : mAttentionType(attentionType)
            , mKvFactor(kvFactor)

        {
        }

        [[nodiscard]] bool operator==(AttentionConfig const& other) const noexcept
        {
            return mAttentionType == other.mAttentionType && mKvFactor == other.mKvFactor;
        }

        // attentionType ;
        AttentionType mAttentionType;
        int mKvFactor;
    };

    [[nodiscard]] ModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] ParallelConfig const& getParallelConfig() const
    {
        return mParallelConfig;
    }

    [[nodiscard]] AttentionConfig const& getAttentionConfig() const
    {
        return mAttentionConfig;
    }

    [[nodiscard]] nvinfer1::DataType const& getDataType() const
    {
        return mDataType;
    }

    [[nodiscard]] bool getEnableBlockReuse() const
    {
        return mEnableBlockReuse;
    }

    [[nodiscard]] bool getHasIndexerKCache() const
    {
        return mHasIndexerKCache;
    }

    [[nodiscard]] SizeType32 getIndexerDimPerHead() const
    {
        return mIndexerDimPerHead;
    }

    [[nodiscard]] SizeType32 getIndexerKCacheQuantBlockSize() const
    {
        return mIndexerKCacheQuantBlockSize;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        sstring << "numKvHeads\n";
        for (auto kvHead : mModelConfig.mNbKvHeadsPerLayer)
        {
            sstring << kvHead << "\n";
        }
        sstring << "sizePerHead:" << mModelConfig.mSizePerHead << "\n";
        sstring << "mTokensPerBlock:" << mModelConfig.mTokensPerBlock << "\n";
        sstring << "tp:" << mParallelConfig.mTensorParallelism << "\n";
        sstring << "pp:" << mParallelConfig.mPipelineParallelism << "\n";
        sstring << "cp:" << mParallelConfig.mContextParallelism << "\n";
        sstring << "enableAttentionDP:" << mParallelConfig.mEnableAttentionDP << "\n";
        sstring << "datatype:" << static_cast<int32_t>(mDataType) << "\n";
        sstring << "attentionType:" << static_cast<int32_t>(mAttentionConfig.mAttentionType) << "\n";
        sstring << "kvFactor:" << mAttentionConfig.mKvFactor << "\n";
        sstring << "dpRank:" << mParallelConfig.mDPrank << "\n";
        sstring << "dpSize:" << mParallelConfig.mDPsize << "\n";
        sstring << "enableBlockReuse:" << mEnableBlockReuse << "\n";
        sstring << "hasIndexerKCache:" << mHasIndexerKCache << "\n";
        sstring << "indexerDimPerHead:" << mIndexerDimPerHead << "\n";
        sstring << "indexerKCacheQuantBlockSize:" << mIndexerKCacheQuantBlockSize << "\n";
        return sstring.str();
    }

private:
    friend class tensorrt_llm::executor::Serialization;
    ModelConfig mModelConfig;
    ParallelConfig mParallelConfig;
    nvinfer1::DataType mDataType;
    AttentionConfig mAttentionConfig;
    bool mEnableBlockReuse{false};
    bool mHasIndexerKCache{false};
    SizeType32 mIndexerDimPerHead{0};
    SizeType32 mIndexerKCacheQuantBlockSize{128};
};

struct MpiState
{
    [[nodiscard]] bool operator==(MpiState const& other) const noexcept
    {
        return mRanks == other.mRanks;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        sstring << "[";
        for (auto rank : mRanks)
        {
            sstring << rank << ",";
        }
        sstring << "]";
        return sstring.str();
    }

    std::vector<SizeType32> mRanks;
};

struct SocketState
{
    [[nodiscard]] bool operator==(SocketState const& other) const noexcept
    {
        return mPort == other.mPort && mIp == other.mIp;
    }

    [[nodiscard]] std::string toString() const
    {
        return mIp + ":" + std::to_string(mPort);
    }

    std::uint16_t mPort;
    std::string mIp;
};

struct AgentState
{
    AgentState(std::string agentName, std::string connectionInfo)
        : mAgentName(std::move(agentName))
        , mConnectionInfo(std::move(connectionInfo))
    {
    }

    AgentState() = default;

    [[nodiscard]] bool operator==(AgentState const& other) const noexcept
    {
        return mAgentName == other.mAgentName && mConnectionInfo == other.mConnectionInfo;
    }

    [[nodiscard]] std::string toString() const
    {
        return mAgentName;
    }

    std::string mAgentName;
    std::string mConnectionInfo;
};

class CommState final
{
public:
    CommState() = default;

    explicit CommState(std::vector<SizeType32> ranks, int selfIdx = -1)
        : mState{MpiState{std::move(ranks)}}
        , mSelfIdx{selfIdx}
    {
    }

    explicit CommState(std::vector<SocketState> socketState, int selfIdx = -1)
        : mState{std::move(socketState)}
        , mSelfIdx{selfIdx}
    {
    }

    CommState(std::uint16_t port, std::string ip)
        : mState{std::vector<SocketState>{SocketState{port, std::move(ip)}}}
        , mSelfIdx{0}
    {
    }

    explicit CommState(std::vector<AgentState> agentState, int selfIdx = -1)
        : mState{std::move(agentState)}
        , mSelfIdx{selfIdx}
    {
    }

    [[nodiscard]] bool isMpiState() const noexcept
    {
        return std::holds_alternative<MpiState>(mState);
    }

    [[nodiscard]] bool isSocketState() const noexcept
    {
        return std::holds_alternative<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] bool isAgentState() const noexcept
    {
        return std::holds_alternative<std::vector<AgentState>>(mState);
    }

    [[nodiscard]] MpiState const& getMpiState() const
    {
        TLLM_CHECK(isMpiState());
        return std::get<MpiState>(mState);
    }

    [[nodiscard]] std::vector<SocketState> const& getSocketState() const
    {
        TLLM_CHECK(isSocketState());
        return std::get<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] std::vector<AgentState> const& getAgentState() const
    {
        TLLM_CHECK(isAgentState());
        return std::get<std::vector<AgentState>>(mState);
    }

    [[nodiscard]] int getSelfIdx() const noexcept
    {
        return mSelfIdx;
    }

    [[nodiscard]] bool operator==(CommState const& other) const noexcept
    {
        return mState == other.mState;
    }

    [[nodiscard]] std::string toString() const
    {
        if (isMpiState())
        {
            return "MPI:" + getMpiState().toString();
        }
        if (isSocketState())
        {
            std::stringstream sstring;
            sstring << "SOCKET:[";
            for (auto&& socket : getSocketState())
            {
                sstring << socket.toString() << ",";
            }
            sstring << "]";
            return sstring.str();
        }
        if (isAgentState())
        {
            std::stringstream sstring;
            sstring << "AGENT:[";
            for (auto&& agent : getAgentState())
            {
                sstring << agent.toString() << ",";
            }
            sstring << "]";
            return sstring.str();
        }

        return "";
    }

private:
    friend class tensorrt_llm::executor::Serialization;
    std::variant<std::monostate, MpiState, std::vector<SocketState>, std::vector<AgentState>> mState;
    int mSelfIdx{-1};
};

} // namespace kv_cache

class DataTransceiverState final
{
public:
    DataTransceiverState() = default;

    DataTransceiverState(kv_cache::CacheState cacheState, kv_cache::CommState commState)
        : mCacheState{std::move(cacheState)}
        , mCommState{std::move(commState)}
    {
    }

    void setCacheState(kv_cache::CacheState state)
    {
        mCacheState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CacheState> const& getCacheState() const noexcept
    {
        return mCacheState;
    }

    void setCommState(kv_cache::CommState state)
    {
        mCommState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CommState> const& getCommState() const noexcept
    {
        return mCommState;
    }

    [[nodiscard]] bool operator==(DataTransceiverState const& other) const noexcept
    {
        return mCacheState == other.mCacheState && mCommState == other.mCommState;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        if (mCacheState)
        {
            sstring << mCacheState.value().toString();
        }
        if (mCommState)
        {
            sstring << mCommState.value().toString();
        }
        return sstring.str();
    }

private:
    friend class Serialization;
    std::optional<kv_cache::CacheState> mCacheState;
    std::optional<kv_cache::CommState> mCommState;
};

inline std::string getLocalIpByNic(std::string const& interface, int rank)
{
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1)
    {
        TLLM_LOG_ERROR(rank,
            "getLocalIpByNic: Can't get local ip from NIC Interface. Please check whether TRTLLM_UCX_INTERFACE is set "
            "correctly.");
        return std::string{};
    }

    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr == nullptr)
        {
            continue;
        }

        if (ifa->ifa_name == interface)
        {
            if (ifa->ifa_addr->sa_family == AF_INET)
            {
                char ip[INET_ADDRSTRLEN]{};
                void* addr = &((reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr);
                if ((inet_ntop(AF_INET, addr, ip, sizeof(ip)) != nullptr) && std::strcmp(ip, "0.0.0.0") != 0)
                {
                    freeifaddrs(ifaddr);
                    return std::string(ip);
                }
            }
            else if (ifa->ifa_addr->sa_family == AF_INET6)
            {
                char ip[INET6_ADDRSTRLEN]{};
                void* addr = &((reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr))->sin6_addr);
                if ((inet_ntop(AF_INET6, addr, ip, sizeof(ip)) != nullptr) && std::strncmp(ip, "fe80::", 6) != 0
                    && std::strcmp(ip, "::1") != 0)
                {
                    freeifaddrs(ifaddr);
                    return std::string(ip);
                }
            }
        }
    }

    freeifaddrs(ifaddr);
    TLLM_LOG_ERROR(
        rank, "Can't get local ip from NIC Interface. Please check whether TRTLLM_UCX_INTERFACE is set correctly.");
    return std::string{};
}

inline std::string getLocalIpByHostname(int rank)
{
    char hostname[256]{};
    if (gethostname(hostname, sizeof(hostname)) == -1)
    {
        TLLM_LOG_ERROR(rank, "getLocalIpByHostname: Can't get hostname");
        return std::string{};
    }

    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_CANONNAME;

    struct addrinfo* res = nullptr;
    if (getaddrinfo(hostname, nullptr, &hints, &res) != 0)
    {
        TLLM_LOG_WARNING(rank, "getLocalIpByHostname: Can't get address info for hostname");
        return std::string{};
    }

    for (struct addrinfo* p = res; p != nullptr; p = p->ai_next)
    {

        if (p->ai_family == AF_INET)
        { // IPv4
            char ip[INET_ADDRSTRLEN]{};
            struct sockaddr_in* ipv4 = reinterpret_cast<struct sockaddr_in*>(p->ai_addr);
            void* addr = &(ipv4->sin_addr);
            if ((inet_ntop(AF_INET, addr, ip, sizeof(ip)) != nullptr) && std::strcmp(ip, "127.0.0.1") != 0
                && std::strcmp(ip, "0.0.0.0") != 0)
            {
                freeaddrinfo(res);
                return std::string(ip);
            }
        }
        else if (p->ai_family == AF_INET6)
        { // IPv6
            char ip[INET6_ADDRSTRLEN]{};
            struct sockaddr_in6* ipv6 = reinterpret_cast<struct sockaddr_in6*>(p->ai_addr);
            void* addr = &(ipv6->sin6_addr);
            if ((inet_ntop(AF_INET6, addr, ip, sizeof(ip)) != nullptr) && std::strncmp(ip, "fe80::", 6) != 0
                && std::strcmp(ip, "::1") != 0)
            {
                freeaddrinfo(res);
                return std::string(ip);
            }
        }
    }

    freeaddrinfo(res);
    TLLM_LOG_WARNING(rank, "getLocalIpByHostname: Can't get local ip from hostname");
    return std::string{};
}

inline std::string getLocalIpByRemoteOrHostName(int rank)
{
    // Try IPv4
    struct sockaddr_in addr
    {
    };

    addr.sin_family = AF_INET;
    addr.sin_port = htons(80);
    // using google's public dns server to get the local ip which can be accessed from remote
    char const* dns_ip_v4 = "8.8.8.8";
    inet_pton(AF_INET, dns_ip_v4, &addr.sin_addr);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock != -1)
    {
        if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != -1)
        {
            socklen_t addr_len = sizeof(addr);
            if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) != -1)
            {
                char ip[INET_ADDRSTRLEN]{};
                inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
                close(sock);
                return std::string(ip);
            }
        }
        close(sock);
    }

    // Try IPv6
    struct sockaddr_in6 addr6
    {
    };

    addr6.sin6_family = AF_INET6;
    addr6.sin6_port = htons(80);
    // using google's public dns server
    char const* dns_ipv6 = "2001:4860:4860::8888";
    inet_pton(AF_INET6, dns_ipv6, &addr6.sin6_addr);

    sock = socket(AF_INET6, SOCK_DGRAM, 0);
    if (sock != -1)
    {
        if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr6), sizeof(addr6)) != -1)
        {
            socklen_t addr_len = sizeof(addr6);
            if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr6), &addr_len) != -1)
            {
                char ip[INET6_ADDRSTRLEN]{};
                inet_ntop(AF_INET6, &addr6.sin6_addr, ip, sizeof(ip));
                close(sock);
                return std::string(ip);
            }
        }
        close(sock);
    }

    // Try hostname
    return getLocalIpByHostname(rank);
}

inline std::string getLocalIp(int rank)
{
    std::string ucxInterface = tensorrt_llm::common::getEnvUCXInterface();
    std::string localIP = {};
    if (!ucxInterface.empty())
    {
        localIP = getLocalIpByNic(ucxInterface, rank);
    }
    if (localIP.empty())
    {
        localIP = getLocalIpByRemoteOrHostName(rank);
    }
    // check whether the localIP is valid
    if (localIP.empty())
    {
        TLLM_THROW("getLocalIp: Can't get local ip");
    }
    return localIP;
}

} // namespace tensorrt_llm::executor
