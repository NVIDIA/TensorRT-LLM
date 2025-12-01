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

#pragma once

#include "nixl.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include <atomic>
#include <thread>

namespace tensorrt_llm::executor::kv_cache
{

struct NixlHelper
{
    [[nodiscard]] static nixl_mem_t convert(MemoryType type);
    [[nodiscard]] static nixlBasicDesc convert(MemoryDesc const& desc);
    [[nodiscard]] static nixl_reg_dlist_t convertRegDlist(RegisterDescs const& descs);
    [[nodiscard]] static nixl_reg_dlist_t convertRegDlist(FileDescs const& descs);
    [[nodiscard]] static nixl_xfer_op_t convert(TransferOp const& op);
    [[nodiscard]] static nixl_xfer_dlist_t convertXferDist(TransferDescs const& descs);
    [[nodiscard]] static nixl_xfer_dlist_t convertXferDist(FileDescs const& descs);
    static void posixGpuToFileFallback(MemoryDescs const& memoryDesc, FileDescs const& fileDescs);
    static void posixFileToGpuFallback(MemoryDescs const& memoryDesc, FileDescs const& fileDescs);
};

class NixlTransferStatus final : public TransferStatus
{
public:
    NixlTransferStatus(nixlAgent* agent, nixlXferReqH* handle);

    [[nodiscard]] bool isCompleted() const override;

    void wait() const override;

private:
    nixlAgent* mRawAgent{};
    nixlXferReqH* mHandle{};
};

class NixlTransferAgent final : public BaseTransferAgent
{
public:
    NixlTransferAgent(BaseAgentConfig const& config);
    ~NixlTransferAgent();

    void registerMemory(RegisterDescs const& descs) override;

    void deregisterMemory(RegisterDescs const& descs) override;

    void loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc) override;

    AgentDesc getLocalAgentDesc() override;

    void invalidateRemoteAgent(std::string const& name) override;

    [[nodiscard]] std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) override;

    [[nodiscard]] nixlAgent* getRawAgent() const noexcept
    {
        return mRawAgent.get();
    }

    nixl_opt_args_t* getExtraParams() noexcept
    {
        return &mExtraParams;
    }

    void notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) override;

    [[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> getNotifiedSyncMessages() override;

    ConnectionInfoType getLocalConnectionInfo() override;

    void loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) override;

    bool checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs) override;

private:
    std::unique_ptr<nixlAgent> mRawAgent;
    nixlBackendH* mRawBackend{};
    nixl_opt_args_t mExtraParams;
    std::string mName;
    std::string mAddress;

    std::vector<char> mDRamSrcBuffer;
    std::vector<char> mDRamDstBuffer;
};

class NixlLoopbackAgent final : public BaseLoopbackAgent
{
public:
    NixlLoopbackAgent(BaseAgentConfig const& config);
    virtual ~NixlLoopbackAgent() = default;

    virtual void executeLoopbackRequest(
        MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload) override;

private:
    int registerMemory(MemoryDescs const& descs);
    int deregisterMemory(MemoryDescs const& descs);
    int registerFiles(FileDescs const& descs);
    int deregisterFiles(FileDescs const& descs);

    [[nodiscard]] std::unique_ptr<TransferStatus> submitLoopbackRequests(
        MemoryDescs const& memoryDescs, FileDescs const& filedescs, bool isOffload);

    std::unique_ptr<nixlAgent> mRawAgent;
    std::string mName;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    [[nodiscard]] std::unique_ptr<BaseTransferAgent> createNixlTransferAgent(BaseAgentConfig const* config);
}

extern "C"
{
    [[nodiscard]] std::shared_ptr<BaseLoopbackAgent> createNixlLoopbackAgent(BaseAgentConfig const* config);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
