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

// How a remote cache level's pages move, apart from the memory it claims (remoteMem.h). A RemoteTransferBackend
// reaches the donors' chunks and moves pages; each submission returns a RemoteTransferAttempt whose outcome and
// quiescence are separate, as in _torch/disaggregation/base/backend.py's Attempt: memory a submission may still
// access (the local staging range, the remote chunk) is reused only once quiesced() is true.
//
// Backends: remoteTransferAgent.h (any executor::kv_cache::BaseTransferAgent, NIXL by default) and
// remoteTransferFabric.h (fabric memory mapped into this process; why it is not an agent is said there).

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! `timeoutMs` from now on the steady clock, clamped to the clock's range: a timeout meant as "never" (e.g.
//! transfer_timeout_s 1e10) must not wrap the deadline into the past.
[[nodiscard]] inline std::chrono::steady_clock::time_point remoteDeadlineAfter(int64_t timeoutMs) noexcept
{
    auto const now = std::chrono::steady_clock::now();
    auto const maxMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::time_point::max() - now)
              .count();
    return now + std::chrono::milliseconds(std::clamp<int64_t>(timeoutMs, 0, maxMs));
}

//! A donor of one rendezvous directory, as RemoteMemClient read it from the donor's files.
struct RemoteDonor
{
    std::string name;
    std::string host;
    uint64_t chunkBytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> segments; // (base, bytes)
    std::vector<std::pair<uint64_t, uint64_t>> chunks;   // (address, bytes), in claim-index order; fabric: offsets
    std::string payload;    // the donor's payloadSuffix() file: what the backend needs to reach it
    bool attached = false;  // the backend has reached it (attachChunk)
    bool withdrawn = false; // its .ready file is gone: no new claims from it
    size_t nextClaim = 0;   // chunks before it were claimed (by anyone) when last probed: claims start here
    bool full = false;      // a probe found every chunk claimed; skipped until a claim finds no donor at all
    //! Per chunk: its claim file was in the last directory listing, or a create since found it taken. Such a chunk is
    //! not probed again until the next listing shows it free.
    std::vector<uint8_t> claimed;
};

//! One page to move: `bytes` between local memory at `local` and donor `donor` at `remoteAddr`.
struct RemoteIo
{
    MemAddress local;
    uint32_t donor;
    uint64_t remoteAddr;
};

//! How one submission ended: delivered whole, or why not. Says nothing about whether access has ended.
struct RemoteTransferOutcome
{
    std::string fault; // empty: every page delivered

    [[nodiscard]] bool ok() const noexcept
    {
        return fault.empty();
    }
};

//! The handle of one submission. Outcome (poll) and quiescence (quiesced) are independent: a submission that timed
//! out on the caller's deadline has no outcome yet and may still access its memory, which then stays out of use.
//! Destroying an attempt that is not quiesced leaks what the backend needs to finish it rather than release it.
class RemoteTransferAttempt
{
public:
    virtual ~RemoteTransferAttempt() = default;

    //! nullopt until concluded; the same outcome on every call after. Never blocks.
    [[nodiscard]] virtual std::optional<RemoteTransferOutcome> poll() = 0;

    //! True once the backend has evidence that no part of the submission still accesses its local or remote
    //! memory; never turns false again. Never blocks.
    [[nodiscard]] virtual bool quiesced() = 0;

    //! poll() until concluded or `timeoutMs` passed (the caller's deadline): nullopt at the deadline.
    [[nodiscard]] virtual std::optional<RemoteTransferOutcome> wait(int64_t timeoutMs) = 0;
};

//! Moves pages between local memory and donated chunks; one per RemoteMemClient, which serializes its calls.
class RemoteTransferBackend
{
public:
    virtual ~RemoteTransferBackend() = default;

    //! The transport's name, as donors write it in their .meta ("nixl" when absent).
    [[nodiscard]] virtual char const* kind() const noexcept = 0;
    //! The suffix of the donor file holding what this backend needs to reach a donor.
    [[nodiscard]] virtual char const* payloadSuffix() const noexcept = 0;
    //! Whether `donor`'s payload fits its chunks; logs why not.
    [[nodiscard]] virtual bool acceptPayload(RemoteDonor const& donor) const = 0;
    //! Make the newly claimed chunk `chunk` of `donor` reachable; returns the address transfers name it by. Throws
    //! RemoteMemError when the donor cannot be reached, which passes over that donor; any other exception (a local
    //! failure, such as mapping the chunk) fails the claim.
    virtual uint64_t attachChunk(RemoteDonor& donor, size_t chunk) = 0;

    //! Whether transfers are enqueued on the caller's stream. Then the stream orders everything that touches the
    //! submission's memory, and a stream-owned staging lease is its quiescence. Otherwise a transfer runs on the
    //! host: its local memory must be ready, and free of stream work, before submit().
    [[nodiscard]] virtual bool streamOrdered() const noexcept = 0;

    //! Register local memory that transfers read from or write into, reference-counted by base address. A backend
    //! that reaches local memory without registration ignores it.
    virtual void registerLocal(MemAddress /*base*/, size_t /*bytes*/) {}

    virtual void deregisterLocal(MemAddress /*base*/, size_t /*bytes*/) noexcept {}

    //! Start moving `bytes` for every entry (`ios` non-empty), local -> remote (`write`) or remote -> local. Returns
    //! without waiting on a donor; a failure after any page may have been accessed is the attempt's fault, never a
    //! throw.
    [[nodiscard]] virtual std::unique_ptr<RemoteTransferAttempt> submit(bool write,
        std::vector<RemoteDonor> const& donors, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream)
        = 0;

    //! Wait until every stream-ordered transfer enqueued so far has finished; a failure is logged (teardown).
    virtual void drain() noexcept {}
};

//! The backend `config` selects; `name` names its agent (unique among the process's clients).
[[nodiscard]] std::unique_ptr<RemoteTransferBackend> makeRemoteTransferBackend(
    RemoteTransferConfig const& config, std::string const& name);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
