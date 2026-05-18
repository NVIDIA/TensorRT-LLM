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
"""Request-aware channel wrapper for Draft API transport packets.

Byte-compat port of the izzy reference
(``tensorrt_llm/_torch/speculative/spec_decode_channel.py``).  This layer
is pure Python; the only external dependency is the protocol module from
Phase 1.  An ``EndpointBackend`` Protocol is consumed for actual
send/recv — Phase 2 ships an in-memory backend, Phase 3 swaps in the
libibverbs implementation.

The imm_data layout below is the wire contract with izzy LPU servers:

    bits  [ 0:  8) — message type   (uint8, MessageType enum)
    bits  [ 8: 20) — stream id      (uint12)
    bits  [20: 32) — slot id        (uint12)

Slot is what makes WRITE_WITH_IMM useful here: the recv buffer pool on
the far side is indexed by slot, so the receiver knows which request
the packet belongs to without reading the payload.
"""

import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from .draft_api_protocol import DraftApiProtocol

_MSG_TYPE_MASK = 0xFF
_STREAM_MASK = 0x0FFF
_SLOT_MASK = 0x0FFF
_STREAM_SHIFT = 8
_SLOT_SHIFT = 20


class EndpointStatus(IntEnum):
    """Generic endpoint statuses used by the channel abstraction."""

    kOk = 0
    kQueueFull = 1
    kEmpty = 2
    kNotStarted = 3
    kNoRecvCredits = 4
    kInvalidPayload = 5
    kUnsupportedBackend = 6
    kError = 7


@dataclass
class EndpointPacket:
    imm_data: int = 0
    payload: bytes = b""


class Endpoint(Protocol):
    def start(self): ...
    def stop(self): ...
    def prime_recv(self, count): ...
    def poll_once(self): ...
    def send(self, packet): ...
    def recv(self): ...


class SpecDecodeChannel:
    """Route-aware packet channel on top of endpoint send/recv semantics."""

    class Status(IntEnum):
        kOk = 0
        kEndpointQueueFull = 1
        kEndpointEmpty = 2
        kEndpointNotStarted = 3
        kEndpointNoRecvCredits = 4
        kEndpointUnsupported = 5
        kEndpointError = 6
        kProtocolError = 7
        kInvalidImmData = 8
        kRouteNotFound = 9
        kRouteAlreadyBound = 10
        kRouteInUse = 11
        kRouteOutOfRange = 12

    @dataclass(frozen=True)
    class Route:
        stream_id: int = 0
        slot: int = 0

    @dataclass(frozen=True)
    class ImmData:
        msg_type: int = 0
        stream_id: int = 0
        slot: int = 0

    @dataclass
    class ReceivedMessage:
        imm_data: "SpecDecodeChannel.ImmData"
        message: DraftApiProtocol.Message

    kImmStreamMax = 0x0FFF
    kImmSlotMax = 0x0FFF

    def __init__(self, endpoint, validation_options=None):
        self._endpoint = endpoint
        self._validation_options = validation_options or DraftApiProtocol.ValidationOptions()
        self._route_lock = threading.Lock()
        self._request_routes = {}
        self._route_to_request = {}

    @staticmethod
    def _coerce_endpoint_status(status):
        if isinstance(status, EndpointStatus):
            return status

        if isinstance(status, IntEnum):
            try:
                return EndpointStatus(int(status))
            except ValueError:
                pass

        if isinstance(status, int):
            try:
                return EndpointStatus(status)
            except ValueError:
                pass

        name = getattr(status, "name", None)
        if isinstance(name, str):
            lowered = name.lower()
        elif isinstance(status, str):
            lowered = status.lower()
        else:
            lowered = ""
        lowered = "".join(ch for ch in lowered if ch.isalnum())

        mapping = {
            "kok": EndpointStatus.kOk,
            "ok": EndpointStatus.kOk,
            "kqueuefull": EndpointStatus.kQueueFull,
            "queuefull": EndpointStatus.kQueueFull,
            "kempty": EndpointStatus.kEmpty,
            "empty": EndpointStatus.kEmpty,
            "knotstarted": EndpointStatus.kNotStarted,
            "notstarted": EndpointStatus.kNotStarted,
            "knorecvcredits": EndpointStatus.kNoRecvCredits,
            "norecvcredits": EndpointStatus.kNoRecvCredits,
            "kinvalidpayload": EndpointStatus.kInvalidPayload,
            "invalidpayload": EndpointStatus.kInvalidPayload,
            "kunsupportedbackend": EndpointStatus.kUnsupportedBackend,
            "unsupportedbackend": EndpointStatus.kUnsupportedBackend,
        }
        return mapping.get(lowered, EndpointStatus.kError)

    @staticmethod
    def _map_endpoint_status(status):
        endpoint_status = SpecDecodeChannel._coerce_endpoint_status(status)
        if endpoint_status == EndpointStatus.kOk:
            return SpecDecodeChannel.Status.kOk
        if endpoint_status == EndpointStatus.kQueueFull:
            return SpecDecodeChannel.Status.kEndpointQueueFull
        if endpoint_status == EndpointStatus.kEmpty:
            return SpecDecodeChannel.Status.kEndpointEmpty
        if endpoint_status == EndpointStatus.kNotStarted:
            return SpecDecodeChannel.Status.kEndpointNotStarted
        if endpoint_status == EndpointStatus.kNoRecvCredits:
            return SpecDecodeChannel.Status.kEndpointNoRecvCredits
        if endpoint_status == EndpointStatus.kUnsupportedBackend:
            return SpecDecodeChannel.Status.kEndpointUnsupported
        return SpecDecodeChannel.Status.kEndpointError

    @staticmethod
    def _validate_route(route):
        if (
            route.stream_id < 0
            or route.slot < 0
            or route.stream_id > SpecDecodeChannel.kImmStreamMax
            or route.slot > SpecDecodeChannel.kImmSlotMax
        ):
            return SpecDecodeChannel.Status.kRouteOutOfRange
        return SpecDecodeChannel.Status.kOk

    @staticmethod
    def _route_key(route):
        return (int(route.stream_id) << 16) | int(route.slot)

    @staticmethod
    def pack_imm_data(fields):
        if (
            fields.stream_id < 0
            or fields.slot < 0
            or fields.stream_id > SpecDecodeChannel.kImmStreamMax
            or fields.slot > SpecDecodeChannel.kImmSlotMax
            or fields.msg_type < 0
            or fields.msg_type > _MSG_TYPE_MASK
        ):
            return SpecDecodeChannel.Status.kInvalidImmData, None

        raw = int(fields.msg_type)
        raw |= int(fields.stream_id) << _STREAM_SHIFT
        raw |= int(fields.slot) << _SLOT_SHIFT
        return SpecDecodeChannel.Status.kOk, raw

    @staticmethod
    def unpack_imm_data(imm_data):
        fields = SpecDecodeChannel.ImmData(
            msg_type=int(imm_data) & _MSG_TYPE_MASK,
            stream_id=(int(imm_data) >> _STREAM_SHIFT) & _STREAM_MASK,
            slot=(int(imm_data) >> _SLOT_SHIFT) & _SLOT_MASK,
        )
        return SpecDecodeChannel.Status.kOk, fields

    def start(self):
        return SpecDecodeChannel._map_endpoint_status(self._endpoint.start())

    def stop(self):
        return SpecDecodeChannel._map_endpoint_status(self._endpoint.stop())

    def prime_recv(self, count):
        return SpecDecodeChannel._map_endpoint_status(self._endpoint.prime_recv(count))

    def send(self, route, msg_type, message):
        route_status = SpecDecodeChannel._validate_route(route)
        if route_status != SpecDecodeChannel.Status.kOk:
            return route_status

        protocol_status, payload = DraftApiProtocol.encode(message, self._validation_options)
        if protocol_status != DraftApiProtocol.Status.kOk:
            return SpecDecodeChannel.Status.kProtocolError

        imm_status, imm_data = SpecDecodeChannel.pack_imm_data(
            SpecDecodeChannel.ImmData(msg_type=msg_type, stream_id=route.stream_id, slot=route.slot)
        )
        if imm_status != SpecDecodeChannel.Status.kOk or imm_data is None:
            return imm_status

        packet = EndpointPacket(imm_data=imm_data, payload=payload)
        return SpecDecodeChannel._map_endpoint_status(self._endpoint.send(packet))

    def send_for_request(self, request_id, msg_type, message):
        route_status, route = self.route_for_request(request_id)
        if route_status != SpecDecodeChannel.Status.kOk or route is None:
            return route_status
        return self.send(route, msg_type, message)

    def recv(self):
        endpoint_status, packet = self._endpoint.recv()
        mapped_status = SpecDecodeChannel._map_endpoint_status(endpoint_status)
        if mapped_status != SpecDecodeChannel.Status.kOk:
            return mapped_status, None
        if packet is None:
            return SpecDecodeChannel.Status.kEndpointError, None

        imm_status, imm_data = SpecDecodeChannel.unpack_imm_data(packet.imm_data)
        if imm_status != SpecDecodeChannel.Status.kOk:
            return imm_status, None

        protocol_status, message = DraftApiProtocol.decode(packet.payload, self._validation_options)
        if protocol_status != DraftApiProtocol.Status.kOk or message is None:
            return SpecDecodeChannel.Status.kProtocolError, None

        return SpecDecodeChannel.Status.kOk, SpecDecodeChannel.ReceivedMessage(
            imm_data=imm_data, message=message
        )

    def pump_once(self):
        poll_status = SpecDecodeChannel._map_endpoint_status(self._endpoint.poll_once())
        if poll_status != SpecDecodeChannel.Status.kOk:
            return poll_status, None

        recv_status, received = self.recv()
        if recv_status != SpecDecodeChannel.Status.kOk:
            return recv_status, None
        assert received is not None

        # Refill exactly one credit so the recv pool stays stable; treat
        # ``kEndpointQueueFull`` as benign (caller already at the cap).
        refill_status = self.prime_recv(1)
        if refill_status in (
            SpecDecodeChannel.Status.kOk,
            SpecDecodeChannel.Status.kEndpointQueueFull,
        ):
            return SpecDecodeChannel.Status.kOk, received
        return refill_status, None

    def pump_once_for_bound_request(self):
        status, received = self.pump_once()
        if status != SpecDecodeChannel.Status.kOk or received is None:
            return status, None, received

        route = SpecDecodeChannel.Route(
            stream_id=received.imm_data.stream_id,
            slot=received.imm_data.slot,
        )
        route_status, request_id = self.request_for_route(route)
        if route_status != SpecDecodeChannel.Status.kOk:
            return route_status, None, received
        return SpecDecodeChannel.Status.kOk, request_id, received

    def bind_request_route(self, request_id, route):
        route_status = SpecDecodeChannel._validate_route(route)
        if route_status != SpecDecodeChannel.Status.kOk:
            return route_status

        with self._route_lock:
            key = SpecDecodeChannel._route_key(route)
            existing_for_route = self._route_to_request.get(key)
            if existing_for_route is not None and existing_for_route != request_id:
                return SpecDecodeChannel.Status.kRouteInUse

            existing_route = self._request_routes.get(request_id)
            if existing_route is not None:
                if existing_route == route:
                    return SpecDecodeChannel.Status.kOk
                return SpecDecodeChannel.Status.kRouteAlreadyBound

            self._request_routes[request_id] = route
            self._route_to_request[key] = request_id
            return SpecDecodeChannel.Status.kOk

    def route_for_request(self, request_id):
        with self._route_lock:
            route = self._request_routes.get(request_id)
            if route is None:
                return SpecDecodeChannel.Status.kRouteNotFound, None
            return SpecDecodeChannel.Status.kOk, route

    def request_for_route(self, route):
        route_status = SpecDecodeChannel._validate_route(route)
        if route_status != SpecDecodeChannel.Status.kOk:
            return route_status, None

        with self._route_lock:
            request_id = self._route_to_request.get(SpecDecodeChannel._route_key(route))
            if request_id is None:
                return SpecDecodeChannel.Status.kRouteNotFound, None
            return SpecDecodeChannel.Status.kOk, request_id

    def unbind_request_route(self, request_id):
        with self._route_lock:
            route = self._request_routes.get(request_id)
            if route is None:
                return SpecDecodeChannel.Status.kRouteNotFound
            del self._request_routes[request_id]
            self._route_to_request.pop(SpecDecodeChannel._route_key(route), None)
            return SpecDecodeChannel.Status.kOk

    @staticmethod
    def to_string(status):
        try:
            parsed = SpecDecodeChannel.Status(int(status))
        except (TypeError, ValueError):
            return "kUnknown"

        if parsed == SpecDecodeChannel.Status.kOk:
            return "kOk"
        if parsed == SpecDecodeChannel.Status.kEndpointQueueFull:
            return "kEndpointQueueFull"
        if parsed == SpecDecodeChannel.Status.kEndpointEmpty:
            return "kEndpointEmpty"
        if parsed == SpecDecodeChannel.Status.kEndpointNotStarted:
            return "kEndpointNotStarted"
        if parsed == SpecDecodeChannel.Status.kEndpointNoRecvCredits:
            return "kEndpointNoRecvCredits"
        if parsed == SpecDecodeChannel.Status.kEndpointUnsupported:
            return "kEndpointUnsupported"
        if parsed == SpecDecodeChannel.Status.kEndpointError:
            return "kEndpointError"
        if parsed == SpecDecodeChannel.Status.kProtocolError:
            return "kProtocolError"
        if parsed == SpecDecodeChannel.Status.kInvalidImmData:
            return "kInvalidImmData"
        if parsed == SpecDecodeChannel.Status.kRouteNotFound:
            return "kRouteNotFound"
        if parsed == SpecDecodeChannel.Status.kRouteAlreadyBound:
            return "kRouteAlreadyBound"
        if parsed == SpecDecodeChannel.Status.kRouteInUse:
            return "kRouteInUse"
        if parsed == SpecDecodeChannel.Status.kRouteOutOfRange:
            return "kRouteOutOfRange"
        return "kUnknown"
