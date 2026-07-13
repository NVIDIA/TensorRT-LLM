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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ControlChannel.h"

#include <deque>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

class nixlAgent;

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// NixlNotifControlChannel — control plane over NIXL notifications (genNotif/getNotifs)
// ----------------------------------------------------------------------------
// Carries the bounce control messages over the SAME fabric the data plane uses (UCX active
// messages, typically RDMA-backed) instead of a dedicated ZMQ/TCP socket pair. Motivation: with the
// scatter plan compressed to a handful of runs, the remaining ackWait cost is per-hop latency of
// SMALL control messages — a ZMQ/TCP hop costs tens of microseconds while a UCX AM hop is a few.
//
// Reverse-path bootstrap: the disagg metadata exchange is one-directional (the KV sender loads the
// receiver's agent metadata, never vice versa), so the receiver initially CANNOT genNotif back.
// localEndpoint() therefore returns this agent's serialized NIXL metadata; the WANT carries it and
// the receiver's onWant self-bootstrap (addPeer) does loadRemoteMD — after which GRANT/ACK flow
// back over the fabric. Mirrors the ZMQ channel's endpoint-in-WANT bootstrap, so mixed channel
// types across peers are NOT supported: enable TRTLLM_NIXL_BOUNCE_NIXL_CONTROL on both sides.
//
// Notification ownership: this channel drains the agent's getNotifs() queue, which is a
// process-wide singleton per agent. Blobs without the bounce magic are dropped (with a warning) —
// the PYTHON transceiver runtime never consumes agent notifications, but the C++ runtime's
// DataSender does, so this channel must NOT be enabled under transceiver_runtime=CPP.
//
// recv() poll model: getNotifs() is a non-blocking poll (no fd to select on), so recv() spins
// getNotifs with a short sleep until the deadline. The reactor already calls recv(timeout 0) when
// busy; only the idle 1ms timeout turns into a sleep-poll loop (bounded, low rate).
// ============================================================================
class NixlNotifControlChannel : public ControlChannel
{
public:
    /// @param agent the (thread-safe) NIXL agent shared with the data plane; borrowed, not owned.
    NixlNotifControlChannel(nixlAgent* agent, std::string selfName);

    [[nodiscard]] std::string localEndpoint() const override;
    void addPeer(std::string const& peer, std::string const& endpoint) override;
    void removePeer(std::string const& peer) override;
    void sendTo(std::string const& peer, std::string const& blob) override;
    [[nodiscard]] bool recv(std::string& outPeer, std::string& outBlob, int timeoutMs) override;

private:
    /// Drain the agent's notification queue into mInbox (bounce-magic blobs only). Returns true if
    /// anything (bounce or not) was drained. Caller must NOT hold mMu.
    bool drainNotifs();

    nixlAgent* mAgent; // codespell:ignore
    std::string mSelfName;

    // Local agent metadata (the WANT-carried "endpoint"), fetched lazily on first use — the arena
    // is registered AFTER the channel is constructed, and conn info is stable afterwards.
    mutable std::mutex mMdMu;
    mutable std::string mLocalMd;

    std::mutex mMu;                                         // guards mPeers + mInbox
    std::unordered_set<std::string> mPeers;
    std::deque<std::pair<std::string, std::string>> mInbox; // (peer, blob), FIFO
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
