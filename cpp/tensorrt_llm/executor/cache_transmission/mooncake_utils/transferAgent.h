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

#include <atomic>
#include <mutex>
#include <thread>

#include "tensorrt_llm/executor/transferAgent.h"
#include "transfer_engine_c.h"

namespace tensorrt_llm::executor::kv_cache
{
class MooncakeTransferAgent final : public BaseTransferAgent
{
public:
    MooncakeTransferAgent(BaseAgentConfig const& config);
    ~MooncakeTransferAgent();

    void registerMemory(RegisterDescs const& descs) override;

    void deregisterMemory(RegisterDescs const& descs) override;

    void loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc) override;

    AgentDesc getLocalAgentDesc() override;

    void invalidateRemoteAgent(std::string const& name) override;

    [[nodiscard]] std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) override;

    /*
     [[nodiscard]] TransferEngine* getRawAgent() const noexcept
    {
        return mRawAgent.get();
    }

    nixl_opt_args_t* getExtraParams() noexcept
    {
        return &mExtraParams;
    }
    */

    void notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) override;

    [[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> getNotifiedSyncMessages() override;

    ConnectionInfoType getConnectionInfo() override;

    void connectRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) override;

    bool checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs) override;

private:
    struct AgentInfo
    {
        int segment_id;
    };

    mutable std::mutex mutex_;
    transfer_engine_t engine_;
    std::string local_agent_name_;
    std::string segment_name_;
};
} // namespace tensorrt_llm::executor::kv_cache
