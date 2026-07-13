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

#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// TransferEngine — abstract bulk-write transport for the bounce data plane
// ----------------------------------------------------------------------------
// Role
//   The ONE operation the bounce data plane needs: "write `bytes` from my local slot
//   buffer to a peer's slot buffer, and let me know (by polling) when it has landed
//   at the remote". The bounce reactor gates the DATA control message on poll()==Done,
//   which (for the real NIXL engine) already implies remote landing via NIXL's
//   ucp_ep_flush_nbx — so v2 needs NO notifMsg.
//
// Why abstract
//   - NixlTransferEngine (production + tests): wraps nixlAgent registerMem / createXferReq /
//     postXferReq / getXferStatus. The bounce tests drive the full pipeline over REAL NIXL RDMA.
//   - Tests can also substitute a trivial fault-injecting engine (one whose poll() always reports
//     kFailed) to exercise the reactor's transfer-failure path deterministically — something a real
//     NIXL engine cannot be coerced into.
//   The reactor is written against this interface only, so the production data path and the
//   failure-injection test share one code path.
//
// Threading
//   Used only by the IO-reactor thread (post + poll + release). Implementations need
//   not be internally locked for cross-thread use.
// ============================================================================

enum class XferState
{
    kInProgress,
    kDone,
    kFailed
};

class TransferEngine
{
public:
    virtual ~TransferEngine() = default;

    /// Register a local region (the bounce arena) so it may be an RDMA src/dst. Returns false if
    /// registration failed so the caller can fall back cleanly (rather than proceeding unregistered
    /// and surfacing the failure later as per-request postWrite errors). Always true (no-op) for the
    /// local-copy engine.
    [[nodiscard]] virtual bool registerRegion(void* addr, std::size_t bytes) = 0;

    /// Post a write of `bytes` from local `src` to `peer`'s device address `dstAddr` on the peer's
    /// GPU `remoteDevId` (carried in the credit; do NOT assume it equals the local device index).
    /// `stream` is advisory and may be null: the reactor posts a write only AFTER the gather event
    /// has signalled (postXferReq is not stream-ordered anyway), so implementations need no stream
    /// ordering. Returns an opaque handle to poll. The handle owns no slot — slot lifetime is the
    /// caller's.
    [[nodiscard]] virtual std::uint64_t postWrite(std::string const& peer, void const* src, std::uint64_t dstAddr,
        std::uint32_t remoteDevId, std::uint32_t bytes, cudaStream_t stream)
        = 0;

    /// Poll a posted write. kDone implies the data has landed (and is visible) at the remote.
    [[nodiscard]] virtual XferState poll(std::uint64_t handle) = 0;

    /// Release handle resources after it reaches a terminal state.
    virtual void release(std::uint64_t handle) = 0;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
