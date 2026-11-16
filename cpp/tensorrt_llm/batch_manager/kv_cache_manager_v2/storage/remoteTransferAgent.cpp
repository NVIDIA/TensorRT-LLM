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

#include "kv_cache_manager_v2/storage/remoteTransferAgent.h"
#include "kv_cache_manager_v2/exceptions.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <algorithm>
#include <chrono>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{

//! The requests of one submission, one per donor.
class AgentTransferAttempt final : public RemoteTransferAttempt
{
public:
    struct Request
    {
        std::string donor; // "<name> (<host>)"
        size_t pages = 0;
        std::unique_ptr<kvc::TransferStatus> status;
        bool concluded = false;
    };

    //! Room for `numRequests` requests and a fault message, so that add() and failSubmit() do not allocate.
    AgentTransferAttempt(std::string opName, size_t bytes, size_t numRequests)
        : mOpName(std::move(opName))
        , mBytes(bytes)
    {
        mRequests.reserve(numRequests);
        mFault.reserve(kFaultCapacity);
    }

    //! Take a posted request; never throws, within the reserved room.
    void add(std::string donor, size_t pages, std::unique_ptr<kvc::TransferStatus> status) noexcept
    {
        TLLM_CHECK_DEBUG(mRequests.size() < mRequests.capacity());
        mRequests.push_back(Request{std::move(donor), pages, std::move(status), /*concluded=*/false});
    }

    //! Posting the request to `donor` failed; nothing more is posted. Never throws.
    void failSubmit(std::string const& donor, char const* what) noexcept
    {
        try
        {
            mFault = "KVCM2 remote level: " + mOpName + " donor " + donor + ": submitting failed: " + what;
        }
        catch (...)
        {
            mFault.assign("KVCM2 remote level: submitting a transfer failed"); // within the reserved capacity
        }
    }

    ~AgentTransferAttempt() override
    {
        if (quiesced())
        {
            return;
        }
        // Releasing a request in progress does not stop it: leave it to run, and its memory out of use.
        size_t leaked = 0;
        for (auto& request : mRequests)
        {
            if (!request.concluded)
            {
                static_cast<void>(request.status.release());
                ++leaked;
            }
        }
        TLLM_LOG_ERROR(
            "KVCM2 remote level: %zu %s request(s) still in progress are left running", leaked, mOpName.c_str());
    }

    AgentTransferAttempt(AgentTransferAttempt const&) = delete;
    AgentTransferAttempt& operator=(AgentTransferAttempt const&) = delete;

    std::optional<RemoteTransferOutcome> poll() override
    {
        advance();
        return outcome();
    }

    //! A request that ended in kFAILURE counts as concluded, its access to the staging range ended: NIXL reports
    //! no state after it, and keeping its lease out of use would leak the claims and staging of every dead donor.
    bool quiesced() override
    {
        advance();
        return allConcluded();
    }

    std::optional<RemoteTransferOutcome> wait(int64_t timeoutMs) override
    {
        auto const deadline = remoteDeadlineAfter(timeoutMs);
        for (auto& request : mRequests)
        {
            if (!request.concluded)
            {
                auto const remaining = std::max<int64_t>(0,
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())
                        .count());
                conclude(request, request.status->wait(remaining));
            }
        }
        return outcome();
    }

private:
    //! Query every request still in progress once, without waiting.
    void advance()
    {
        for (auto& request : mRequests)
        {
            if (!request.concluded)
            {
                conclude(request, request.status->wait(0));
            }
        }
    }

    void conclude(Request& request, kvc::TransferState state)
    {
        if (state == kvc::TransferState::kIN_PROGRESS)
        {
            return;
        }
        request.concluded = true;
        if (state == kvc::TransferState::kFAILURE && mFault.empty())
        {
            mFault = "KVCM2 remote level: " + mOpName + " donor " + request.donor + " of "
                + std::to_string(request.pages) + " page(s) x " + std::to_string(mBytes) + " B failed ("
                + request.status->getLastStatusStr() + ")";
        }
    }

    [[nodiscard]] bool allConcluded() const
    {
        return std::all_of(mRequests.begin(), mRequests.end(), [](Request const& r) { return r.concluded; });
    }

    [[nodiscard]] std::optional<RemoteTransferOutcome> outcome() const
    {
        if (!allConcluded())
        {
            return std::nullopt;
        }
        return RemoteTransferOutcome{mFault};
    }

    std::string mOpName;
    size_t mBytes;
    std::vector<Request> mRequests;
    std::string mFault; // the first failure: the submission's, else a request's

    static constexpr size_t kFaultCapacity = 512;
};

} // namespace

AgentTransferBackend::AgentTransferBackend(std::unique_ptr<kvc::BaseTransferAgent> agent, std::string kind)
    : mTransferAgent(std::move(agent))
    , mKind(std::move(kind))
{
    TLLM_CHECK_WITH_INFO(mTransferAgent != nullptr, "KVCM2 remote level: creating the transfer agent failed");
}

AgentTransferBackend::~AgentTransferBackend()
{
    try
    {
        for (auto const& [base, info] : mLocalRegions)
        {
            mTransferAgent->deregisterMemory(
                kvc::MemoryDescs(kvc::MemoryType::kDRAM, {kvc::MemoryDesc(base, info.first, /*deviceId=*/0)}));
        }
        for (auto const& donor : mLoadedDonors)
        {
            mTransferAgent->invalidateRemoteAgent(donor);
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("KVCM2 remote level: transfer agent teardown: %s", e.what());
    }
}

uint64_t AgentTransferBackend::attachChunk(RemoteDonor& donor, size_t chunk)
{
    if (!donor.attached)
    {
        try
        {
            mTransferAgent->loadRemoteAgent(donor.name, kvc::AgentDesc::deserialize(donor.payload));
        }
        catch (std::exception const& e)
        {
            throw RemoteMemError(
                "KVCM2 remote level: loading donor " + donor.name + " (" + donor.host + ") failed: " + e.what());
        }
        donor.attached = true;
        mLoadedDonors.push_back(donor.name);
    }
    return donor.chunks[chunk].first;
}

void AgentTransferBackend::registerLocal(MemAddress base, size_t bytes)
{
    auto it = mLocalRegions.find(base);
    if (it != mLocalRegions.end())
    {
        TLLM_CHECK_WITH_INFO(it->second.first == bytes,
            "KVCM2 remote level: local region re-registered with %zu B, was %zu B", bytes, it->second.first);
        ++it->second.second;
        return;
    }
    mTransferAgent->registerMemory(
        kvc::MemoryDescs(kvc::MemoryType::kDRAM, {kvc::MemoryDesc(base, bytes, /*deviceId=*/0)}));
    mLocalRegions.emplace(base, std::make_pair(bytes, 1));
}

void AgentTransferBackend::deregisterLocal(MemAddress base, size_t bytes) noexcept
{
    auto it = mLocalRegions.find(base);
    if (it == mLocalRegions.end() || --it->second.second > 0)
    {
        return;
    }
    try
    {
        mTransferAgent->deregisterMemory(
            kvc::MemoryDescs(kvc::MemoryType::kDRAM, {kvc::MemoryDesc(base, bytes, /*deviceId=*/0)}));
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("KVCM2 remote level: deregistering local memory failed: %s", e.what());
    }
    mLocalRegions.erase(it);
}

std::unique_ptr<RemoteTransferAttempt> AgentTransferBackend::submit(bool write, std::vector<RemoteDonor> const& donors,
    std::vector<RemoteIo> const& ios, size_t bytes, CUstream /*stream*/)
{
    std::vector<std::vector<size_t>> byDonor(donors.size());
    for (size_t i = 0; i < ios.size(); ++i)
    {
        TLLM_CHECK_DEBUG(ios[i].donor < donors.size());
        byDonor[ios[i].donor].push_back(i);
    }

    // Everything that allocates is made before the first request is posted: from then on submit must return the
    // attempt that owns the requests in flight, never throw (dropping a status does not stop its request).
    struct Pending
    {
        std::string donor; // "<name> (<host>)"
        size_t pages;
        kvc::TransferRequest request;
    };

    std::vector<Pending> pending;
    for (uint32_t donorIdx = 0; donorIdx < byDonor.size(); ++donorIdx)
    {
        auto const& entries = byDonor[donorIdx];
        if (entries.empty())
        {
            continue;
        }
        std::vector<kvc::MemoryDesc> local;
        std::vector<kvc::MemoryDesc> remote;
        local.reserve(entries.size());
        remote.reserve(entries.size());
        for (size_t i : entries)
        {
            local.emplace_back(ios[i].local, bytes, /*deviceId=*/0);
            remote.emplace_back(static_cast<uintptr_t>(ios[i].remoteAddr), bytes, /*deviceId=*/0);
        }
        auto const& donor = donors[donorIdx];
        // The request names the local descriptors first and the remote ones second, whichever way data flows.
        pending.push_back(Pending{donor.name + " (" + donor.host + ")", entries.size(),
            kvc::TransferRequest(write ? kvc::TransferOp::kWRITE : kvc::TransferOp::kREAD,
                kvc::MemoryDescs(kvc::MemoryType::kDRAM, std::move(local)),
                kvc::MemoryDescs(kvc::MemoryType::kDRAM, std::move(remote)), donor.name)});
    }
    auto attempt = std::make_unique<AgentTransferAttempt>(write ? "WRITE to" : "READ from", bytes, pending.size());
    for (auto& entry : pending)
    {
        std::unique_ptr<kvc::TransferStatus> status;
        try
        {
            status = mTransferAgent->submitTransferRequests(entry.request);
        }
        catch (std::exception const& e)
        {
            // Nothing more is submitted; the requests already in flight conclude before the attempt does.
            attempt->failSubmit(entry.donor, e.what());
            break;
        }
        attempt->add(std::move(entry.donor), entry.pages, std::move(status));
    }
    return attempt;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
