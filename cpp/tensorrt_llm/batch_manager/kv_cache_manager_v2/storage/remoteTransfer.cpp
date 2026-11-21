/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kv_cache_manager_v2/storage/remoteTransfer.h"
#include "kv_cache_manager_v2/storage/remoteTransferAgent.h"
#include "kv_cache_manager_v2/storage/remoteTransferFabric.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/transferAgent.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

std::unique_ptr<RemoteTransferBackend> makeRemoteTransferBackend(
    RemoteTransferConfig const& config, std::string const& name)
{
    switch (config.transport)
    {
    case RemoteTransport::kNixl:
    {
        executor::kv_cache::BaseAgentConfig agentConfig{};
        agentConfig.mName = name;
        // The progress thread completes connection set-up with donors that never call into NIXL themselves.
        agentConfig.useProgThread = true;
        agentConfig.multiThread = false;
        agentConfig.useListenThread = false;
        agentConfig.enableTelemetry = false;
        // This agent's own devices, so the process-wide UCX_NET_DEVICES of the KV transceiver's agent stays.
        agentConfig.backendParams.insert(config.backendParams.begin(), config.backendParams.end());
        return std::make_unique<AgentTransferBackend>(
            executor::kv_cache::makeTransferAgent("nixl", &agentConfig), "nixl");
    }
    case RemoteTransport::kFabric: return std::make_unique<FabricTransferBackend>();
    }
    TLLM_THROW("KVCM2 remote level: unknown transport %d", static_cast<int>(config.transport));
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
