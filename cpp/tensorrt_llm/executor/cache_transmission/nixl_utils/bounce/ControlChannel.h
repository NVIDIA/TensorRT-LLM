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

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// ControlChannel — bounce v2 control-plane transport
// ----------------------------------------------------------------------------
// Role
//   Carries the small control messages of the credit protocol between agents:
//   WANT / GRANT / DATA / ACK (encoded by BounceMessage). It is a named-peer,
//   asynchronous message channel: sendTo(peer, blob) is fire-and-forget and recv()
//   pulls the next inbound (peer, blob). The capability handshake is exchanged
//   out-of-band in AgentDesc, not as a ControlChannel message. WANT still carries
//   the sender's endpoint so the receiver can self-bootstrap the reverse path;
//   an empty WANT cancels a request (there is no separate RETURN message).
//
// Why it is abstract
//   The data plane (bulk RDMA) is always NIXL. The control plane is pluggable so
//   we can pick the transport without touching the reactor / credit logic:
//     - ZmqControlChannel        (default) — dedicated, event-driven zmq sockets.
//     - NixlNotifControlChannel — reuse NIXL genNotif/getNotifs (no extra port).
//   Crucially, NEITHER uses NIXL's postXferReq notifMsg — v2 abandons it entirely
//   (data-landed is detected by the sender polling getXferStatus, which already
//   includes NIXL's ucp_ep_flush_nbx).
//
// Ordering / correctness contract
//   The channel need NOT order messages w.r.t. the RDMA data plane: the only
//   message with a data dependency is DATA ("go scatter"), and the sender emits
//   DATA *after* observing getXferStatus==SUCCESS, so the data is already landed
//   at the remote regardless of which transport carries DATA. Messages sent from
//   one peer MUST be delivered FIFO: GRANT entries are associated with chunks by
//   order and do not carry an explicit chunk index.
//
// Threading contract
//   recv() is called only by the single IO-reactor thread. sendTo()/addPeer() may
//   be called concurrently (reactor, app threads issuing WANT, and scatter workers
//   issuing ACK). Implementations must make sendTo()/addPeer() mutually thread-safe.
//   Only the reactor calls recv(), so implementations need not support concurrent
//   receivers.
// ============================================================================
class ControlChannel
{
public:
    virtual ~ControlChannel() = default;

    /// The endpoint other agents need to reach us: a ZMQ address or serialized NIXL metadata,
    /// depending on the implementation. Advertised in the AgentDesc capability handshake and in
    /// WANT for reverse-path bootstrap.
    [[nodiscard]] virtual std::string localEndpoint() const = 0;

    /// Register where to reach `peer` (its localEndpoint()). Must be called before
    /// the first sendTo(peer, ...). Idempotent.
    virtual void addPeer(std::string const& peer, std::string const& endpoint) = 0;

    /// Forget `peer`: drop its send-side socket/endpoint (e.g. when the peer is lost, so the
    /// per-peer resource isn't kept alive until shutdown). Idempotent; thread-safe with
    /// sendTo()/addPeer(). A sendTo() to a removed peer is dropped until it is re-added — either
    /// explicitly via addPeer() or implicitly by the WANT self-bootstrap in the receiver's onWant.
    virtual void removePeer(std::string const& peer) = 0;

    /// Fire-and-forget send of an encoded BounceMessage `blob` to `peer`.
    /// Thread-safe with other sendTo()/addPeer() calls.
    virtual void sendTo(std::string const& peer, std::string const& blob) = 0;

    /// Block up to `timeoutMs` for one inbound message. On arrival, fills `outPeer`
    /// (sender agent name) and `outBlob` (encoded message) and returns true;
    /// returns false on timeout. Reactor-thread only.
    [[nodiscard]] virtual bool recv(std::string& outPeer, std::string& outBlob, int timeoutMs) = 0;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
