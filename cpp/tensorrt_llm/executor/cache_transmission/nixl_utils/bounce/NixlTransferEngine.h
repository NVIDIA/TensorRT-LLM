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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/TransferEngine.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

class nixlAgent; // NIXL, global namespace

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// NixlTransferEngine — production TransferEngine backed by a nixlAgent (DESIGN)
// ----------------------------------------------------------------------------
// Wraps a borrowed nixlAgent. registerRegion() -> registerMem (VRAM); postWrite() ->
// createXferReq(NIXL_WRITE, [src,bytes] -> [dstAddr,bytes] @ peer) + postXferReq;
// poll() -> getXferStatus; release() -> releaseXferReq. NO notifMsg is ever attached
// (DESIGN.md §4): getXferStatus==SUCCESS already implies the data has landed at the
// remote target because NIXL's UCX backend appends a ucp_ep_flush_nbx to every transfer.
//
// The dst address comes from a credit (the receiver's registered region address); NIXL
// resolves it against the remote agent's metadata (exchanged via loadRemoteAgent). The
// caller (BounceTransport) guarantees the gather kernel has completed before postWrite,
// since NIXL is not ordered against the gather's CUDA stream.
//
// The remote (dst) GPU index is taken from the credit (BounceCreditEntry::devId, set by the
// receiver) — NOT assumed equal to the local device id — so writes are correct even when
// sender/receiver don't share a device index. The local (src) desc uses this engine's mDeviceId.
// ============================================================================
class NixlTransferEngine : public TransferEngine
{
public:
    NixlTransferEngine(nixlAgent* agent, int deviceId);
    ~NixlTransferEngine() override;

    [[nodiscard]] bool registerRegion(void* addr, std::size_t bytes) override;
    [[nodiscard]] std::uint64_t postWrite(std::string const& peer, void const* src, std::uint64_t dstAddr,
        std::uint32_t remoteDevId, std::uint32_t bytes, cudaStream_t stream) override;
    [[nodiscard]] XferState poll(std::uint64_t handle) override;
    void release(std::uint64_t handle) override;

private:
    nixlAgent* mRawAgent{nullptr};
    int mDeviceId{0};
    std::mutex mMu;
    std::uint64_t mNext{1};
    std::unordered_map<std::uint64_t, void*> mHandles;      // id -> nixlXferReqH*
    std::vector<std::pair<void*, std::size_t>> mRegistered; // (addr,bytes) registered -> deregister in dtor
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
