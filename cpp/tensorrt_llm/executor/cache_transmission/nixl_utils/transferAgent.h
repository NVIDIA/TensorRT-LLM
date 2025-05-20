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
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/interfaces.h"
#include "tensorrt_llm/executor/transferAgent.h"

namespace tensorrt_llm::executor::kv_cache
{

struct NixlHelper
{
    [[nodiscard]] static nixl_mem_t convert(MemoryType type);
    [[nodiscard]] static nixlBasicDesc convert(MemoryDesc const& desc);
    [[nodiscard]] static nixl_reg_dlist_t convertRegDlist(RegisterDescs const& descs);
    [[nodiscard]] static nixl_xfer_op_t convert(TransferOp const& op);
    [[nodiscard]] static nixl_xfer_dlist_t convertXferDist(TransferDescs const& descs);
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
    NixlTransferAgent(BaseAgentConfig const& config, AgentRegistrar* registrar);

    void registerMemory(RegisterDescs const& descs) override;

    void deregisterMemory(RegisterDescs const& descs) override;

    void loadRemoteAgent(char const* name) override;

    void invalidateRemoteAgent(char const* name) override;

    [[nodiscard]] std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) override;

    [[nodiscard]] nixlAgent* getRawAgent() const noexcept
    {
        return mRawAgent.get();
    }

    nixl_opt_args_t* getExtraParams() noexcept
    {
        return &mExtraParams;
    }

private:
    std::unique_ptr<nixlAgent> mRawAgent;
    nixlBackendH* mRawBackend{};
    AgentRegistrar* mRegistrar{};
    nixl_opt_args_t mExtraParams;
    std::string mName;
};

} // namespace tensorrt_llm::executor::kv_cache
