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
#include <memory>
#include <mutex>
#include <string>
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

using TransferDescs = MemoryDescs;
using RegisterDescs = MemoryDescs;

class AgentDesc final
{
public:
    AgentDesc(std::vector<char> backendAgentDesc)
        : mBackendAgentDesc{backendAgentDesc}
    {
    }

    [[nodiscard]] std::vector<char> const& getBackendAgentDesc() const noexcept
    {
        return mBackendAgentDesc;
    }

private:
    std::vector<char> mBackendAgentDesc;
};

enum class TransferOp : uint8_t
{
    kREAD,
    kWRITE,
};

class TransferRequest
{
public:
    TransferRequest(TransferOp op, TransferDescs srcDescs, TransferDescs dstDescs, char const* remoteName)
        : mOp{op}
        , mSrcDescs{std::move(srcDescs)}
        , mDstDescs{std::move(dstDescs)}
        , mRemoteName{remoteName}
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

    [[nodiscard]] char const* getRemoteName() const noexcept
    {
        return mRemoteName.c_str();
    }

private:
    TransferOp mOp;
    TransferDescs mSrcDescs;
    TransferDescs mDstDescs;
    std::string mRemoteName;
};

class TransferStatus
{
public:
    virtual ~TransferStatus() = default;
    [[nodiscard]] virtual bool isCompleted() const = 0;
    virtual void wait() const = 0;
};

class AgentRegistrar
{
public:
    ~AgentRegistrar() = default;

    [[nodiscard]] virtual AgentDesc const* getAgentDesc(char const* agentName) const = 0;

    virtual void addAgentDesc(char const* agentName, AgentDesc desc) = 0;

    virtual void removeAgentDesc(char const* agentName) = 0;
};

struct BaseAgentConfig
{
    char const* mName;
    bool useProgThread;
};

class BaseTransferAgent
{
public:
    virtual ~BaseTransferAgent() = default;

    virtual void registerMemory(RegisterDescs const& descs) = 0;

    virtual void deregisterMemory(RegisterDescs const& descs) = 0;

    virtual void loadRemoteAgent(char const* name) = 0;

    virtual void invalidateRemoteAgent(char const* name) = 0;

    [[nodiscard]] virtual std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) = 0;

    // TODO: Add `notifySyncInfo` and `getMatchedSyncInfo` interfaces.
};

class DynLibLoader final
{
public:
    [[nodiscard]] static DynLibLoader& getInstance();

    [[nodiscard]] void* getHandle(char const* name);

    template <typename FunctionT>
    [[nodiscard]] FunctionT getFunctionPointer(char const* libName, char const* funcName)
    {
        void* handle = getHandle(libName);
        void* funcPtr = dlSym(handle, funcName);
        const std::string err = funcName + std::string{" function is not open correctly."};
        TLLM_CHECK_WITH_INFO(funcPtr, "%s", err.c_str());
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
[[nodiscard]] std::unique_ptr<BaseTransferAgent> makeTransferAgent(char const* const& backend, Args&&... args)
{
    if (backend == std::string{"nixl"})
    {
        auto& loader = DynLibLoader::getInstance();
        using CreateNixlFuncType = std::unique_ptr<BaseTransferAgent> (*)(BaseAgentConfig const*, AgentRegistrar*);
        auto* func = loader.getFunctionPointer<CreateNixlFuncType>(
            "libtensorrt_llm_nixl_wrapper.so", "createNixlTransferAgent");
        return func(std::forward<Args>(args)...);
    }
    TLLM_THROW("Unknown backend name.");
}

} // namespace tensorrt_llm::executor::kv_cache
