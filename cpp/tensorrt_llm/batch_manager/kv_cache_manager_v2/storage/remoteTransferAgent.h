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

#pragma once

#include "kv_cache_manager_v2/storage/remoteTransfer.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace tensorrt_llm::executor::kv_cache
{
class BaseTransferAgent;
} // namespace tensorrt_llm::executor::kv_cache

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Transfers through a transfer agent (executor::kv_cache::BaseTransferAgent; makeRemoteTransferBackend makes a NIXL
//! one): one-sided READ / WRITE of DRAM, concluded on the host. Donors publish their agent's AgentDesc in
//! donor.<name>.desc. Only the BaseTransferAgent interface is used, so any agent that moves DRAM serves.
//!
//! A request is never released while in progress: releasing one does not stop it (NIXL 1.5.2 frees the backend
//! handle without cancelling or draining it), so an attempt holds its requests until they conclude, and one destroyed
//! before that leaks them. The owner of an attempt that has not quiesced keeps this backend alive too
//! (RemoteMemClient::abandon).
class AgentTransferBackend final : public RemoteTransferBackend
{
public:
    //! `kind` is the transport name donors of this agent write in their .meta.
    AgentTransferBackend(std::unique_ptr<executor::kv_cache::BaseTransferAgent> agent, std::string kind);
    //! Deregisters local memory, forgets the donors and destroys the agent.
    ~AgentTransferBackend() override;

    AgentTransferBackend(AgentTransferBackend const&) = delete;
    AgentTransferBackend& operator=(AgentTransferBackend const&) = delete;

    [[nodiscard]] char const* kind() const noexcept override
    {
        return mKind.c_str();
    }

    [[nodiscard]] char const* payloadSuffix() const noexcept override
    {
        return "desc";
    }

    [[nodiscard]] bool acceptPayload(RemoteDonor const& /*donor*/) const override
    {
        return true;
    }

    uint64_t attachChunk(RemoteDonor& donor, size_t chunk) override;

    [[nodiscard]] bool streamOrdered() const noexcept override
    {
        return false;
    }

    void registerLocal(MemAddress base, size_t bytes) override;
    void deregisterLocal(MemAddress base, size_t bytes) noexcept override;

    //! One request per donor. A donor whose request cannot be submitted is the attempt's fault, which concludes once
    //! the requests already submitted have.
    [[nodiscard]] std::unique_ptr<RemoteTransferAttempt> submit(bool write, std::vector<RemoteDonor> const& donors,
        std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream) override;

private:
    std::unique_ptr<executor::kv_cache::BaseTransferAgent> mTransferAgent;
    std::string mKind;
    std::vector<std::string> mLoadedDonors;
    std::unordered_map<MemAddress, std::pair<size_t, int>> mLocalRegions; // base -> (bytes, refs)
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
