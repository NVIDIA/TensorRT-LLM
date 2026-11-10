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

"""The load record on the wire: ``EngineLoad`` batches and ``GetLoads`` answers.

A router that routes by load subscribes to ``SubscribeKvEvents`` with ``loads_only`` and reads
the servicer's ``EngineLoad`` record from ``load_only`` batches (numbered 0, nothing admitted
from them), one per data-parallel rank whenever the record changed or its heartbeat is due;
``GetLoads`` is its fallback poll. :class:`LoadRecordStream` paces the batches from a
:class:`~.load_tracker.LoadRecord` source; the message builders fill the generated messages.

The ``smg-grpc-proto`` package installed beside the engine may predate two wire fields:
``SubscribeKvEventsRequest.loads_only`` (field 3) and ``EngineLoad.prefill_throughput``
(field 22). Protobuf keeps fields a descriptor does not know as unknown fields, so the first is
read from them (:func:`wants_loads_only`) and the second is written as one
(:func:`engine_load_message`): a router whose proto knows the fields reads them either way. The
``GetLoads`` messages exist only in a package whose ``trtllm_service.proto`` declares the RPC;
the servicer registers the handler only then (:func:`serves_get_loads`).
"""

from __future__ import annotations

import asyncio
import struct
import time
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime, timezone
from types import ModuleType
from typing import Any

from .bindings import common_pb2, trtllm_service_pb2, trtllm_service_pb2_grpc
from .load_tracker import HEARTBEAT_BACKOFF, HEARTBEAT_INTERVAL, LOAD_TICK, LoadRecord, RecordPacer

__all__ = [
    "LOADS_ONLY_FIELD_NUMBER",
    "NO_KV_EVENTS",
    "PREFILL_THROUGHPUT_FIELD_NUMBER",
    "LoadRecordStream",
    "engine_load_message",
    "get_loads_response",
    "load_only_batch",
    "scheduler_load_message",
    "serves_get_loads",
    "wants_loads_only",
]

#: ``smg.grpc.common.SubscribeKvEventsRequest.loads_only``.
LOADS_ONLY_FIELD_NUMBER = 3
#: ``smg.grpc.common.EngineLoad.prefill_throughput``.
PREFILL_THROUGHPUT_FIELD_NUMBER = 22
#: What a subscription for KV events (not ``loads_only``) is refused with.
NO_KV_EVENTS = (
    "KV cache events are not relayed by this servicer; subscribe with loads_only for its load "
    "records"
)

_WIRE_VARINT = 0
_WIRE_FIXED64 = 1
_WIRE_LENGTH = 2
_WIRE_FIXED32 = 5


def _field_known(message: Any, name: str) -> bool:
    return name in message.DESCRIPTOR.fields_by_name


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _read_varint(data: bytes, index: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if index >= len(data) or shift > 63:
            raise ValueError("truncated varint")
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7


def _varint_field(data: bytes, number: int) -> int | None:
    """The last value of varint field ``number`` in an encoded message, ``None`` when absent."""
    found = None
    index = 0
    while index < len(data):
        tag, index = _read_varint(data, index)
        field, wire_type = tag >> 3, tag & 0x7
        if wire_type == _WIRE_VARINT:
            value, index = _read_varint(data, index)
            if field == number:
                found = value
        elif wire_type == _WIRE_FIXED64:
            index += 8
        elif wire_type == _WIRE_LENGTH:
            length, index = _read_varint(data, index)
            index += length
        elif wire_type == _WIRE_FIXED32:
            index += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
    return found


def wants_loads_only(request: Any) -> bool:
    """Whether a ``SubscribeKvEventsRequest`` asks for the load records alone.

    Read from the descriptor when the installed proto has the field, from the unknown fields
    otherwise.
    """
    if _field_known(request, "loads_only"):
        return bool(request.loads_only)
    return bool(_varint_field(request.SerializeToString(), LOADS_ONLY_FIELD_NUMBER))


def _set_optional(message: Any, name: str, value: Any) -> None:
    """Set an optional field the installed proto knows; leave it unset for ``None``."""
    if value is not None and _field_known(message, name):
        setattr(message, name, value)


def engine_load_message(
    record: LoadRecord, *, sample: int, load_only: bool = True
) -> common_pb2.EngineLoad:
    """The ``EngineLoad`` for ``record``, numbered ``sample`` on its stream."""
    message = common_pb2.EngineLoad(
        running_requests=record.running_requests,
        waiting_requests=record.waiting_requests,
        token_usage=record.token_usage,
        gen_throughput=record.gen_throughput,
        max_running_requests=record.max_running_requests,
        age_ms=record.age_ms,
        sample=sample,
        load_only=load_only,
    )
    _set_optional(message, "waiting_uncached_tokens", record.waiting_uncached_tokens)
    _set_optional(message, "cache_hit_rate", record.cache_hit_rate)
    _set_optional(message, "num_used_tokens", record.num_used_tokens)
    _set_optional(message, "max_total_num_tokens", record.max_total_num_tokens)
    _set_optional(message, "prefill_pending_tokens", record.prefill_pending_tokens)
    if record.prefill_throughput is not None:
        if _field_known(message, "prefill_throughput"):
            message.prefill_throughput = record.prefill_throughput
        else:
            tag = (PREFILL_THROUGHPUT_FIELD_NUMBER << 3) | _WIRE_FIXED64
            message.MergeFromString(_varint(tag) + struct.pack("<d", record.prefill_throughput))
    return message


def load_only_batch(
    record: LoadRecord, *, dp_rank: int, sample: int, timestamp: float | None = None
) -> common_pb2.KvEventBatch:
    """A ``load_only`` batch: no events, sequence 0, the rank's record."""
    batch = common_pb2.KvEventBatch(
        sequence_number=0,
        timestamp=time.time() if timestamp is None else timestamp,
        dp_rank=dp_rank,
    )
    batch.load.CopyFrom(engine_load_message(record, sample=sample, load_only=True))
    return batch


def serves_get_loads(stub: type = trtllm_service_pb2_grpc.TrtllmServiceServicer) -> bool:
    """Whether the installed ``trtllm_service.proto`` declares ``GetLoads``.

    The generated server registers only the methods the stub declares.
    """
    return hasattr(stub, "GetLoads")


def scheduler_load_message(
    record: LoadRecord, *, dp_rank: int, pb2: ModuleType | None = None
) -> Any:
    """The ``SchedulerLoad`` of a ``GetLoads`` answer for ``record``."""
    pb2 = trtllm_service_pb2 if pb2 is None else pb2
    message = pb2.SchedulerLoad(
        dp_rank=dp_rank,
        num_running_reqs=record.running_requests,
        num_waiting_reqs=record.waiting_requests,
        num_total_reqs=record.running_requests + record.waiting_requests,
        num_used_tokens=record.num_used_tokens or 0,
        max_total_num_tokens=record.max_total_num_tokens or 0,
        token_usage=record.token_usage,
        gen_throughput=record.gen_throughput,
        cache_hit_rate=record.cache_hit_rate or 0.0,
        utilization=record.token_usage,
        max_running_requests=record.max_running_requests,
        num_waiting_uncached_tokens=record.waiting_uncached_tokens or 0,
    )
    _set_optional(message, "prefill_throughput", record.prefill_throughput)
    return message


def get_loads_response(
    records: Sequence[LoadRecord], *, version: str, pb2: ModuleType | None = None
) -> Any:
    """The ``GetLoadsResponse`` listing ``records`` as ranks 0, 1, ..."""
    pb2 = trtllm_service_pb2 if pb2 is None else pb2
    return pb2.GetLoadsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=version,
        dp_rank_count=len(records),
        loads=[
            scheduler_load_message(record, dp_rank=rank, pb2=pb2)
            for rank, record in enumerate(records)
        ],
    )


class LoadRecordStream:
    """The ``load_only`` batches of one subscription, from a record source.

    Every :attr:`load_tick` the source is read; the record goes out when it changed or its
    heartbeat is due (:class:`~.load_tracker.RecordPacer`), the first one at once. The stream
    ends when the source has no record (the servicer is gone) or ``cancelled`` says so.
    ``clock`` and ``sleep`` are monotonic seconds and an async sleep (tests pass their own).

    Args:
        source: The current record, or ``None`` once there is none to report.
        dp_rank: The rank stamped on every batch.
        load_tick: Seconds between two reads of the source.
        heartbeat_interval: Silence after which an unchanged record goes out again.
        heartbeat_backoff: The interval once two heartbeats in a row were unchanged.
        clock: Monotonic seconds.
        sleep: An awaitable sleep of the given seconds.
    """

    def __init__(
        self,
        source: Callable[[], LoadRecord | None],
        *,
        dp_rank: int = 0,
        load_tick: float = LOAD_TICK,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        heartbeat_backoff: float = HEARTBEAT_BACKOFF,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._source = source
        self._dp_rank = dp_rank
        self._load_tick = load_tick
        self._pacer = RecordPacer(
            heartbeat_interval=heartbeat_interval, heartbeat_backoff=heartbeat_backoff
        )
        self._clock = clock
        self._sleep = sleep

    async def batches(
        self, *, cancelled: Callable[[], bool] = lambda: False
    ) -> AsyncIterator[common_pb2.KvEventBatch]:
        """The batches, as they fall due."""
        sample = 0
        while not cancelled():
            now = self._clock()
            record = self._source()
            if record is None:
                return
            if self._pacer.due(record, now):
                sample += 1
                yield load_only_batch(record, dp_rank=self._dp_rank, sample=sample)
            # The next tick a whole period from this one: a late tick does not burst to catch up.
            await self._sleep(max(0.0, now + self._load_tick - self._clock()))
