/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

enum class MemoryType : uint8_t
{
    kDRAM,
    kVRAM,
    kBLK,
    kOBJ,
    kFILE
};

// `MemoryDesc` is used to describe a memory region, which can then be designated
// as the source or destination of read/write operations.
class MemoryDesc
{
public:
    MemoryDesc(std::vector<char> const& vec, uint32_t deviceId = 0)
        : mAddr{reinterpret_cast<uintptr_t>(vec.data())}
        , mLen{vec.size()}
        , mDeviceId{deviceId}
    {
    }

    MemoryDesc(void* addr, size_t len, uint32_t deviceId)
        : mAddr{reinterpret_cast<uintptr_t>(addr)}
        , mLen{len}
        , mDeviceId{deviceId}
    {
    }

    MemoryDesc(uintptr_t addr, size_t len, uint32_t deviceId)
        : mAddr{addr}
        , mLen{len}
        , mDeviceId{deviceId}
    {
    }

    [[nodiscard]] uintptr_t getAddr() const noexcept
    {
        return mAddr;
    }

    [[nodiscard]] size_t getLen() const noexcept
    {
        return mLen;
    }

    [[nodiscard]] uint32_t getDeviceId() const noexcept
    {
        return mDeviceId;
    }

    static void serialize(MemoryDesc const& memoryDesc, std::ostream& os);
    [[nodiscard]] static MemoryDesc deserialize(std::istream& is);
    [[nodiscard]] static size_t serializedSize(MemoryDesc const& memoryDesc);

private:
    uintptr_t mAddr;
    size_t mLen;
    uint32_t mDeviceId;
};

class MemoryDescs
{
public:
    MemoryDescs(MemoryType type, std::vector<MemoryDesc> descs)
        : mType{type}
        , mDescs{std::move(descs)}
    {
    }

    [[nodiscard]] MemoryType getType() const noexcept
    {
        return mType;
    }

    [[nodiscard]] std::vector<MemoryDesc> const& getDescs() const noexcept
    {
        return mDescs;
    }

private:
    MemoryType mType;
    std::vector<MemoryDesc> mDescs;
};

class FileDesc
{
public:
    FileDesc(std::string const& filename, int flags, mode_t mode, size_t len)
        : mLen{len}
    {
        int fd = ::open(filename.c_str(), flags, mode);
        TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' (GDS)", filename.c_str());
        this->fd = fd;
    }

    FileDesc(FileDesc&& other) noexcept
        : fd(other.fd)
        , mLen(other.mLen)
    {
        other.fd = -1;
        other.mLen = 0;
    }

    FileDesc& operator=(FileDesc&& other) noexcept
    {
        if (this != &other)
        {
            if (fd != -1)
                ::close(fd);
            fd = other.fd;
            mLen = other.mLen;
            other.fd = -1;
            other.mLen = 0;
        }
        return *this;
    }

    ~FileDesc()
    {
        if (fd != -1)
            ::close(fd);
    }

    [[nodiscard]] uint64_t getFd() const noexcept
    {
        return fd;
    }

    [[nodiscard]] size_t getLen() const noexcept
    {
        return mLen;
    }

    FileDesc(FileDesc const&) = delete;
    FileDesc& operator=(FileDesc const&) = delete;

private:
    int fd;
    size_t mLen;
};

class FileDescs
{
public:
    FileDescs(std::vector<FileDesc>&& descs)
        : mDescs(std::move(descs))
    {
    }

    [[nodiscard]] std::vector<FileDesc> const& getDescs() const noexcept
    {
        return mDescs;
    }

private:
    std::vector<FileDesc> mDescs;
};

using TransferDescs = MemoryDescs;
using RegisterDescs = MemoryDescs;
using SyncMessage = std::string;
using ConnectionInfoType = std::string;

// `AgentDesc` represents the unique identifier for reading and writing to the agent.
// By accessing this identifier, the backend can establish the correct connection.
class AgentDesc final
{
public:
    AgentDesc(std::string backendAgentDesc)
        : mBackendAgentDesc{std::move(backendAgentDesc)}
    {
    }

    [[nodiscard]] std::string const& getBackendAgentDesc() const noexcept
    {
        return mBackendAgentDesc;
    }

private:
    std::string mBackendAgentDesc;
};

// `TransferOp` is an enumeration that represents the types of transfer operations.
// Currently, it supports two operations: `read` and `write`.
enum class TransferOp : uint8_t
{
    kREAD,
    kWRITE,
};

// `TransferRequest` is used to represent the transfer requests supported by the underlying agent.
class TransferRequest
{
public:
    /// @brief The constructor of `TransferRequest`.
    /// @param op Source data arrangement.
    /// @param srcDescs Description of the source memory region.
    /// @param dstDescs Description of the destination memory region.
    /// @param remoteName Name of the remote counterpart.
    /// @param syncMessage Synchronization information for the end of the transfer.
    TransferRequest(TransferOp op, TransferDescs srcDescs, TransferDescs dstDescs, std::string const& remoteName,
        std::optional<SyncMessage> syncMessage = std::nullopt)
        : mOp{op}
        , mSrcDescs{std::move(srcDescs)}
        , mDstDescs{std::move(dstDescs)}
        , mRemoteName{remoteName}
        , mSyncMessage{std::move(syncMessage)}
    {
    }

    [[nodiscard]] TransferOp getOp() const noexcept
    {
        return mOp;
    }

    [[nodiscard]] TransferDescs const& getSrcDescs() const noexcept
    {
        return mSrcDescs;
    }

    [[nodiscard]] TransferDescs const& getDstDescs() const noexcept
    {
        return mDstDescs;
    }

    [[nodiscard]] std::string const& getRemoteName() const noexcept
    {
        return mRemoteName;
    }

    [[nodiscard]] std::optional<SyncMessage> getSyncMessage() const noexcept
    {
        return mSyncMessage;
    }

private:
    TransferOp mOp;
    TransferDescs mSrcDescs;
    TransferDescs mDstDescs;
    std::string mRemoteName;
    std::optional<SyncMessage> mSyncMessage;
};

// Data structure for checking the status of active transfer operations.
class TransferStatus
{
public:
    virtual ~TransferStatus() = default;
    [[nodiscard]] virtual bool isCompleted() const = 0;
    virtual void wait() const = 0;
};

struct BaseAgentConfig
{
    std::string mName;
    bool useProgThread;
    bool multiThread;
};

class BaseTransferAgent
{
public:
    virtual ~BaseTransferAgent() = default;

    /// @brief Register a memory region.
    /// @param descs Describe the memory regions to be registered.
    virtual void registerMemory(RegisterDescs const& descs) = 0;

    /// @brief Unregister a memory region.
    /// @param descs Describe the memory regions to be unregistered.
    virtual void deregisterMemory(RegisterDescs const& descs) = 0;

    /// @brief Initialize and establish a connection with a remote agent.
    /// @param name Specify the name of the remote agent.
    /// @param agentDesc Provide the necessary communication details for connecting to the remote agent.
    virtual void loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc) = 0;

    /// @brief Initialize and establish a connection with a remote agent.
    /// @param name Specify the name of the remote agent.
    /// @param connectionInfo Provide the necessary communication details for connecting to the remote agent.
    virtual void loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) = 0;

    /// @brief Invalidate a connection with a remote agent.
    /// @param name Specify the name of the remote agent.
    virtual void invalidateRemoteAgent(std::string const& name) = 0;

    /// @brief Fetch the descriptor of the local agent.
    /// @return The descriptor of the local agent.
    virtual AgentDesc getLocalAgentDesc() = 0;

    /// @brief Fetch the descriptor of the local agent.
    /// @return The descriptor of the local agent.
    virtual ConnectionInfoType getLocalConnectionInfo() = 0;

    /// @brief Initiate the transfer by submitting the request.
    /// @param request Specify the transmission request.
    /// @return The status of the requests.
    [[nodiscard]] virtual std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) = 0;

    /// @brief Generate a notification, not bound to a transfer, e.g., for control.
    /// @param name Specify the name of the remote agent to which the information should be sent.
    /// @param syncMessage The data or message intended for synchronization.
    virtual void notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) = 0;

    /// @brief Retrieve notification messages sent by other agents.
    /// @return A mapping from remote agent names to their respective notification messages.
    virtual std::unordered_map<std::string, std::vector<SyncMessage>> getNotifiedSyncMessages() = 0;

    /// @brief Check if metadata is available for a remote agent.
    /// @return Whether the metadata is available for a remote agent.
    virtual bool checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs) = 0;
};

class BaseLoopbackAgent
{
public:
    virtual ~BaseLoopbackAgent() = default;
    virtual void executeLoopbackRequest(MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload) = 0;
};

class DynLibLoader final
{
public:
    [[nodiscard]] static DynLibLoader& getInstance();

    [[nodiscard]] void* getHandle(std::string const& name);

    template <typename FunctionT>
    [[nodiscard]] FunctionT getFunctionPointer(std::string const& libName, std::string const& funcName)
    {
        void* handle = getHandle(libName);
        void* funcPtr = dlSym(handle, funcName.c_str());
        TLLM_CHECK_WITH_INFO(funcPtr, funcName + " function is not open correctly.");
        return reinterpret_cast<FunctionT>(funcPtr);
    }

    ~DynLibLoader();

    DynLibLoader() = default;
    DynLibLoader(DynLibLoader const&) = delete;
    DynLibLoader& operator=(DynLibLoader const&) = delete;

private:
    [[nodiscard]] static void* dlSym(void* handle, char const* symbol);

    std::mutex mDllMutex;
    std::unordered_map<std::string, void*> mHandlers;
};

template <typename... Args>
[[nodiscard]] std::unique_ptr<BaseTransferAgent> makeTransferAgent(std::string const& backend, Args&&... args)
{
    if (backend == "nixl")
    {
        auto& loader = DynLibLoader::getInstance();
        using CreateNixlFuncType = std::unique_ptr<BaseTransferAgent> (*)(BaseAgentConfig const*);
        auto* func = loader.getFunctionPointer<CreateNixlFuncType>(
            "libtensorrt_llm_nixl_wrapper.so", "createNixlTransferAgent");
        return func(std::forward<Args>(args)...);
    }
    TLLM_THROW("Unknown backend name.");
}

template <typename... Args>
[[nodiscard]] std::shared_ptr<BaseLoopbackAgent> makeLoopbackAgent(std::string const& backend, Args&&... args)
{
    if (backend == "nixl")
    {
        auto& loader = DynLibLoader::getInstance();
        using CreateNixlFuncType = std::shared_ptr<BaseLoopbackAgent> (*)(BaseAgentConfig const*);
        auto* func = loader.getFunctionPointer<CreateNixlFuncType>(
            "libtensorrt_llm_nixl_wrapper.so", "createNixlLoopbackAgent");
        return func(std::forward<Args>(args)...);
    }
    TLLM_THROW("Unknown backend name.");
}

} // namespace tensorrt_llm::executor::kv_cache
