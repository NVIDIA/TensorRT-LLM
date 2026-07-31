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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
// Reverse-path bootstrap: disaggregated serving supports one-directional metadata exchange (the KV
// sender may load the receiver's agent metadata without the reverse), so the receiver might not be
// able to genNotif back initially. localEndpoint() therefore returns this agent's serialized NIXL
// metadata; the WANT carries it and the receiver's onWant self-bootstrap (addPeer) does loadRemoteMD
// — after which GRANT/ACK flow back over the fabric. This mirrors the ZMQ channel's
// endpoint-in-WANT bootstrap. Mixed channel types across peers are NOT supported: enable
// TRTLLM_NIXL_BOUNCE_USE_NIXL_NOTIFICATIONS on both sides.
//
// Notification ownership: this channel is the sole getNotifs() consumer while enabled because the
// queue is shared by all notification users of that agent. Bounce messages and ordinary sync
// messages are demultiplexed into separate inboxes; takeNonBounceNotifications() exposes the latter
// to NixlTransferAgent::getNotifiedSyncMessages().
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

    /// Drain and return ordinary (non-bounce) agent notifications without exposing bounce control
    /// messages to the transfer-agent sync-message API.
    [[nodiscard]] std::unordered_map<std::string, std::vector<std::string>> takeNonBounceNotifications();

private:
    /// Drain the agent's notification queue into the bounce and non-bounce inboxes. Returns true if
    /// anything was drained. Caller must NOT hold mDrainMu or mMu.
    bool drainNotifs();

    nixlAgent* mAgent; // codespell:ignore
    std::string mSelfName;

    // Local agent metadata (the WANT-carried "endpoint"), fetched lazily on first use — the arena
    // is registered AFTER the channel is constructed, and conn info is stable afterwards.
    mutable std::mutex mMdMu;
    mutable std::string mLocalMd;

    std::mutex mDrainMu; // serializes access to the agent's shared notification queue
    std::mutex mMu;      // guards mPeers and both inboxes
    std::unordered_set<std::string> mPeers;
    std::deque<std::pair<std::string, std::string>> mInbox; // (peer, blob), FIFO
    std::unordered_map<std::string, std::vector<std::string>> mNonBounceInbox;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
