# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for ``SpecDecodeChannel`` and the in-memory backend.

These run without RDMA hardware — the channel sits on top of
``InMemoryEndpointBackend``, which is a deque + recv-credit model.
"""

from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol
from tensorrt_llm._torch.speculative.ibverbs_endpoint import InMemoryEndpointBackend
from tensorrt_llm._torch.speculative.spec_decode_channel import EndpointStatus, SpecDecodeChannel

P = DraftApiProtocol
MT = DraftApiProtocol.MessageType
C = SpecDecodeChannel
CS = SpecDecodeChannel.Status


def _make_target_msg(token=42, round_seq=1, position=0):
    return P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=round_seq,
        position=position,
        num_tokens=1,
        tokens=[token] + [0] * (P.kMaxTokens - 1),
    )


def _make_draft_msg(tokens=(100, 200, 300, 400, 500), round_seq=1, position=0):
    payload = list(tokens) + [0] * (P.kMaxTokens - len(tokens))
    return P.Message(
        message_type=MT.kDraftToTarget,
        round_seq_num=round_seq,
        position=position,
        num_tokens=len(tokens),
        tokens=payload,
    )


def _started_channel():
    backend = InMemoryEndpointBackend(max_queue_depth=4)
    channel = C(backend)
    assert channel.start() == CS.kOk
    return backend, channel


def test_imm_data_pack_unpack_roundtrip():
    s, packed = C.pack_imm_data(C.ImmData(msg_type=1, stream_id=42, slot=99))
    assert s == CS.kOk
    s2, unpacked = C.unpack_imm_data(packed)
    assert s2 == CS.kOk
    assert unpacked == C.ImmData(msg_type=1, stream_id=42, slot=99)


def test_imm_data_packs_into_known_layout():
    """msg_type in low byte, stream in [8:20), slot in [20:32)."""
    s, packed = C.pack_imm_data(C.ImmData(msg_type=0xAB, stream_id=0x123, slot=0x456))
    assert s == CS.kOk
    assert (packed & 0xFF) == 0xAB
    assert ((packed >> 8) & 0x0FFF) == 0x123
    assert ((packed >> 20) & 0x0FFF) == 0x456


def test_imm_data_rejects_out_of_range():
    s, _ = C.pack_imm_data(C.ImmData(msg_type=256, stream_id=0, slot=0))
    assert s == CS.kInvalidImmData
    s, _ = C.pack_imm_data(C.ImmData(msg_type=0, stream_id=0x1000, slot=0))
    assert s == CS.kInvalidImmData
    s, _ = C.pack_imm_data(C.ImmData(msg_type=0, stream_id=0, slot=0x1000))
    assert s == CS.kInvalidImmData


def test_bind_route_unique():
    _, channel = _started_channel()
    assert channel.bind_request_route(7, C.Route(0, 17)) == CS.kOk
    # Different request, same route → kRouteInUse.
    assert channel.bind_request_route(8, C.Route(0, 17)) == CS.kRouteInUse


def test_bind_route_idempotent_for_same_request():
    _, channel = _started_channel()
    assert channel.bind_request_route(7, C.Route(0, 17)) == CS.kOk
    assert channel.bind_request_route(7, C.Route(0, 17)) == CS.kOk


def test_rebind_to_different_route_rejected():
    _, channel = _started_channel()
    assert channel.bind_request_route(7, C.Route(0, 17)) == CS.kOk
    assert channel.bind_request_route(7, C.Route(0, 18)) == CS.kRouteAlreadyBound


def test_unbind_route_clears_both_maps():
    _, channel = _started_channel()
    channel.bind_request_route(7, C.Route(0, 17))
    assert channel.unbind_request_route(7) == CS.kOk
    s, route = channel.route_for_request(7)
    assert s == CS.kRouteNotFound and route is None
    s, req = channel.request_for_route(C.Route(0, 17))
    assert s == CS.kRouteNotFound and req is None
    # And it can be rebound.
    assert channel.bind_request_route(8, C.Route(0, 17)) == CS.kOk


def test_route_validation_out_of_range():
    _, channel = _started_channel()
    assert channel.bind_request_route(1, C.Route(stream_id=0x2000, slot=0)) == CS.kRouteOutOfRange
    assert channel.bind_request_route(1, C.Route(stream_id=0, slot=-1)) == CS.kRouteOutOfRange


def test_send_without_recv_credit_returns_no_credits():
    _, channel = _started_channel()
    channel.bind_request_route(7, C.Route(0, 17))
    s = channel.send_for_request(7, msg_type=int(MT.kTargetToDraft), message=_make_target_msg())
    # No prime_recv was called → backend reports no recv credits.
    assert s == CS.kEndpointNoRecvCredits


def test_full_send_recv_through_inmem_backend():
    _, channel = _started_channel()
    channel.prime_recv(1)
    channel.bind_request_route(7, C.Route(0, 17))

    msg = _make_target_msg(token=5421, round_seq=42, position=100)
    s = channel.send_for_request(7, msg_type=int(MT.kTargetToDraft), message=msg)
    assert s == CS.kOk

    s2, request_id, received = channel.pump_once_for_bound_request()
    assert s2 == CS.kOk
    assert request_id == 7
    assert received.message.tokens[0] == 5421
    assert received.imm_data.msg_type == int(MT.kTargetToDraft)
    assert received.imm_data.stream_id == 0
    assert received.imm_data.slot == 17


def test_pump_when_empty_returns_endpoint_empty():
    _, channel = _started_channel()
    channel.prime_recv(1)
    s, received = channel.pump_once()
    assert s == CS.kEndpointEmpty
    assert received is None


def test_pump_for_bound_request_unknown_route():
    """Receiver gets a packet whose slot isn't bound — should fail at lookup."""
    _, channel = _started_channel()
    channel.prime_recv(1)
    # Bind request 7, but craft a send to a different route that has no binding.
    channel.bind_request_route(7, C.Route(0, 17))
    msg = _make_target_msg()
    # Direct send via raw route bypasses the request-mapping check.
    s = channel.send(C.Route(0, 18), msg_type=int(MT.kTargetToDraft), message=msg)
    assert s == CS.kOk
    s2, request_id, received = channel.pump_once_for_bound_request()
    assert s2 == CS.kRouteNotFound
    assert request_id is None
    # Even on lookup failure, the packet itself was decoded fine.
    assert received is not None and received.message.tokens[0] == 42


def test_send_invalid_protocol_message_returns_protocol_error():
    _, channel = _started_channel()
    channel.prime_recv(1)
    channel.bind_request_route(7, C.Route(0, 17))
    bogus = _make_target_msg()
    bogus.num_tokens = 5  # T2D must be exactly 1 token
    s = channel.send_for_request(7, msg_type=int(MT.kTargetToDraft), message=bogus)
    assert s == CS.kProtocolError


def test_recv_refills_credit_so_pool_stays_stable():
    backend, channel = _started_channel()
    # Pool capacity 4; prime 4, send 1, pump 1 → credits should stay near 4.
    channel.prime_recv(4)
    channel.bind_request_route(7, C.Route(0, 17))
    msg = _make_target_msg()
    assert channel.send_for_request(7, msg_type=int(MT.kTargetToDraft), message=msg) == CS.kOk
    s, _, _ = channel.pump_once_for_bound_request()
    assert s == CS.kOk
    # After pump_once, credit was decremented by send, then incremented by 1
    # via prime_recv inside pump_once → back to 4.
    assert backend._recv_credits == 4


def test_to_string_known_values():
    assert C.to_string(CS.kOk) == "kOk"
    assert C.to_string(CS.kRouteInUse) == "kRouteInUse"
    assert C.to_string("not-an-int") == "kUnknown"


# ---------------------------------------------------------------------------
# InMemoryEndpointBackend low-level checks (independent of the channel).
# ---------------------------------------------------------------------------


def test_inmem_backend_not_started_rejects_ops():
    backend = InMemoryEndpointBackend()
    assert backend.prime_recv(1) == EndpointStatus.kNotStarted
    assert backend.poll_once() == EndpointStatus.kNotStarted
    s, p = backend.recv()
    assert s == EndpointStatus.kNotStarted and p is None


def test_inmem_backend_send_blocks_without_credit():
    backend = InMemoryEndpointBackend(max_queue_depth=2)
    backend.start()
    from tensorrt_llm._torch.speculative.spec_decode_channel import EndpointPacket

    s = backend.send(EndpointPacket(imm_data=0, payload=b"\x00" * 96))
    assert s == EndpointStatus.kNoRecvCredits
    backend.prime_recv(1)
    s = backend.send(EndpointPacket(imm_data=0, payload=b"\x00" * 96))
    assert s == EndpointStatus.kOk


def test_inmem_backend_prime_recv_capped():
    backend = InMemoryEndpointBackend(max_queue_depth=2)
    backend.start()
    assert backend.prime_recv(2) == EndpointStatus.kOk
    assert backend.prime_recv(1) == EndpointStatus.kQueueFull
