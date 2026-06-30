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

#include <zmq.hpp>

#include <mutex>
#include <string>
#include <unordered_map>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// ZmqControlChannel — zmq implementation of ControlChannel (DESIGN)
// ----------------------------------------------------------------------------
// Topology (one-way, symmetric peers)
//   Each agent owns:
//     - ONE ROUTER socket, bound to a tcp endpoint, ZMQ_ROUTING_ID = self name.
//       It only RECEIVES. A peer's DEALER connects here; on recv the ROUTER yields
//       [senderName, blob] (senderName = the peer DEALER's routing id).
//     - ONE DEALER per peer, ZMQ_ROUTING_ID = self name, connected to that peer's
//       ROUTER endpoint. It only SENDS. The peer's ROUTER then sees us by name.
//   So "A -> B" travels A's DEALER(B) -> B's ROUTER; "B -> A" travels B's DEALER(A)
//   -> A's ROUTER. No request/reply coupling — purely async one-way, which is all
//   the credit protocol needs.
//
// Bootstrap
//   localEndpoint() returns the ROUTER's bound address (auto-port via "tcp://host:*"
//   + ZMQ_LAST_ENDPOINT). Endpoints are exchanged out-of-band (NIXL agent metadata)
//   and registered with addPeer() before the first sendTo().
//
// Threading
//   The ROUTER is touched only by the reactor thread (recv). DEALERs (+ the dealer
//   map) are guarded by mMu so app threads and the reactor can sendTo()/addPeer()
//   concurrently. zmq sockets are not thread-safe; the mutex provides the required
//   serialization + memory fence for the dealer sockets, and the ROUTER is never
//   shared. recv() polls the ROUTER with a timeout so the reactor can interleave
//   getXferStatus polling.
// ============================================================================
class ZmqControlChannel : public ControlChannel
{
public:
    /// @param selfName    this agent's name (becomes the zmq routing id).
    /// @param bindAddr    address pattern to bind the ROUTER ("tcp://127.0.0.1:*" by default,
    ///                    "*" picks a free port; query the chosen one via localEndpoint()).
    explicit ZmqControlChannel(std::string selfName, std::string const& bindAddr = "tcp://127.0.0.1:*");
    ~ZmqControlChannel() override;

    [[nodiscard]] std::string localEndpoint() const override;
    void addPeer(std::string const& peer, std::string const& endpoint) override;
    void removePeer(std::string const& peer) override;
    void sendTo(std::string const& peer, std::string const& blob) override;
    [[nodiscard]] bool recv(std::string& outPeer, std::string& outBlob, int timeoutMs) override;

private:
    std::string mSelfName;
    zmq::context_t mCtx;
    zmq::socket_t mRouter;                                   // receive-only (reactor thread)
    std::string mEndpoint;                                   // resolved bound endpoint

    mutable std::mutex mMu;                                  // guards mDealers + dealer sends
    std::unordered_map<std::string, zmq::socket_t> mDealers; // peer -> send-only DEALER
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
