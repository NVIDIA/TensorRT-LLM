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
"""Bounce v2 control-plane wire codec (protocol version 3).

Binary layout compatible with the C++ codec
(cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceMessage.{h,cpp}):
every message is a fixed 44-byte little-endian header followed by a typed
payload. Version 3 changes exactly one thing vs the C++ v2: ACK messages
carry a LIST of ``(chunk_idx, region_handle)`` entries (batched ACK — the
reactor coalesces ACKs per peer per tick, halving the per-chunk event rate on
the sender side); ``header.count`` says how many. A v2 peer rejects v3
messages via the version check and vice versa — mixed-version clusters
cleanly fall back to the standard NIXL path (strict handshake), never
corrupt.

Endianness: fields are packed little-endian explicitly (the C++ memcpy's raw
host order, which is little-endian on all NVIDIA GPU hosts), so both codecs
produce identical bytes.

Decoders NEVER raise on peer input: any malformed/short/foreign blob returns
``None`` (or ``False``), mirroring the C++ bool-returning decoders. ENCODERS,
by contrast, may raise ``struct.error`` on out-of-range integers (e.g. a
negative request id or a value that does not fit its wire field) — encoder
inputs come from local code, so that is a caller bug, not peer input.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Optional, Sequence

import numpy as np

from .plan import SCATTER_RUN_DTYPE

__all__ = [
    "BOUNCE_MAGIC",
    "BOUNCE_VERSION",
    "AckEntry",
    "BounceMsgHeader",
    "BounceMsgType",
    "CreditEntry",
    "decode_ack",
    "decode_credits",
    "decode_header",
    "decode_scatter",
    "decode_want",
    "encode_ack",
    "encode_cancel",
    "encode_data",
    "encode_grant",
    "encode_want",
    "has_bounce_magic",
    "is_cancel_want",
]

BOUNCE_MAGIC = 0x424E4332  # 'B''N''C''2'
#: v3: batched ACK (see module docstring). v2 (C++) is intentionally refused.
BOUNCE_VERSION = 3

# u32 magic, u16 version, u16 msgType, u64 requestId, u64 regionHandle,
# u32 chunkIdx, u32 numChunks, u32 count, u32 payloadBytes, u32 aux
_HEADER = struct.Struct("<IHHQQIIIII")
assert _HEADER.size == 44

# u64 addr, u32 len, u32 devId, u64 regionHandle
_CREDIT = struct.Struct("<QIIQ")
assert _CREDIT.size == 24

# v3 batched-ACK entry: u32 chunkIdx, u64 regionHandle (packed, no padding)
_ACK_ENTRY = struct.Struct("<IQ")
assert _ACK_ENTRY.size == 12

_U32 = struct.Struct("<I")


class BounceMsgType(IntEnum):
    """Control-message kinds (values shared with the C++ enum)."""

    WANT = 1  # sender -> receiver: per-chunk byte sizes (empty list cancels)
    GRANT = 2  # receiver -> sender: credits
    DATA = 3  # sender -> receiver: region written; scatter runs
    ACK = 4  # receiver -> sender: chunks scattered; regions freed (batched)


class BounceMsgHeader(NamedTuple):
    """Decoded fixed header. Field meaning is per-type; unused fields are 0."""

    magic: int
    version: int
    msg_type: int
    request_id: int
    region_handle: int  # DATA: arena region offset of the chunk; 0 elsewhere
    chunk_idx: int  # DATA
    num_chunks: int  # DATA
    count: int  # trailing entries (credits / chunk sizes / scatter / acks)
    payload_bytes: int  # bytes of trailing payload
    aux: int  # WANT: num chunks; 0 elsewhere


@dataclass(frozen=True)
class CreditEntry:
    """Permission to RDMA-write one chunk into a receiver arena allocation.

    ``addr`` is the absolute remote address, ``length`` the chunk's packed
    transfer length (the backing buddy block may be larger), ``dev_id`` the
    RECEIVER's GPU, and ``region_handle`` the receiver arena offset the sender
    echoes in DATA so the receiver can locate and release the allocation.
    """

    addr: int
    length: int
    dev_id: int
    region_handle: int


class AckEntry(NamedTuple):
    """One entry of a v3 batched ACK."""

    chunk_idx: int
    region_handle: int


def _header_bytes(
    msg_type: BounceMsgType,
    request_id: int,
    chunk_idx: int = 0,
    num_chunks: int = 0,
    region_handle: int = 0,
    count: int = 0,
    payload_bytes: int = 0,
    aux: int = 0,
) -> bytes:
    return _HEADER.pack(
        BOUNCE_MAGIC,
        BOUNCE_VERSION,
        int(msg_type),
        request_id,
        region_handle,
        chunk_idx,
        num_chunks,
        count,
        payload_bytes,
        aux,
    )


def encode_want(request_id: int, chunk_sizes: Sequence[int], endpoint: "str | bytes") -> bytes:
    """Encode a WANT: the per-chunk byte sizes the sender will write plus the
    sender's own bounce control endpoint (lets the receiver self-bootstrap the
    reverse control path). An EMPTY size list is a cancel (see
    :func:`encode_cancel`). Payload:
    ``[count * u32 chunk_bytes][u32 endpoint_len][endpoint bytes]``."""
    sizes = np.ascontiguousarray(np.asarray(chunk_sizes, dtype=np.uint32)).reshape(-1)
    ep = endpoint.encode("utf-8") if isinstance(endpoint, str) else bytes(endpoint)
    sizes_blob = sizes.tobytes()
    payload = sizes_blob + _U32.pack(len(ep)) + ep
    header = _header_bytes(
        BounceMsgType.WANT,
        request_id,
        count=sizes.shape[0],
        payload_bytes=len(payload),
        aux=sizes.shape[0],
    )
    return header + payload


def encode_cancel(request_id: int, endpoint: "str | bytes" = b"") -> bytes:
    """Cancel/retract a request: by C++ convention a WANT with an EMPTY chunk
    list — the wire form is identical, so the receiver's onWant/reclaim path
    is reused verbatim. An empty WANT is unambiguously a cancel because a
    0-chunk transfer never sends a WANT at all."""
    return encode_want(request_id, (), endpoint)


def encode_grant(request_id: int, credits: Sequence[CreditEntry]) -> bytes:
    """Encode a GRANT carrying a batch of credits."""
    payload = b"".join(_CREDIT.pack(c.addr, c.length, c.dev_id, c.region_handle) for c in credits)
    header = _header_bytes(
        BounceMsgType.GRANT,
        request_id,
        count=len(credits),
        payload_bytes=len(payload),
    )
    return header + payload


def encode_data(
    request_id: int,
    chunk_idx: int,
    num_chunks: int,
    region_handle: int,
    scatter_runs: np.ndarray,
) -> bytes:
    """Encode a DATA: the region was written; payload is the chunk's scatter
    runs ([m] structured array of ``SCATTER_RUN_DTYPE``, or anything
    convertible to it)."""
    runs = np.ascontiguousarray(np.asarray(scatter_runs, dtype=SCATTER_RUN_DTYPE)).reshape(-1)
    payload = runs.tobytes()
    header = _header_bytes(
        BounceMsgType.DATA,
        request_id,
        chunk_idx=chunk_idx,
        num_chunks=num_chunks,
        region_handle=region_handle,
        count=runs.shape[0],
        payload_bytes=len(payload),
    )
    return header + payload


def encode_ack(request_id: int, entries: Sequence["AckEntry | tuple[int, int]"]) -> bytes:
    """Encode a v3 batched ACK: ``entries`` is a list of
    ``(chunk_idx, region_handle)`` pairs, each meaning that chunk finished
    scattering and its region was freed. The header's per-chunk fields stay 0;
    ``count`` carries the batch size."""
    payload = b"".join(_ACK_ENTRY.pack(int(c), int(r)) for c, r in entries)
    header = _header_bytes(
        BounceMsgType.ACK,
        request_id,
        count=len(entries),
        payload_bytes=len(payload),
    )
    return header + payload


def has_bounce_magic(blob: bytes) -> bool:
    """Lightweight prefix check: does ``blob`` start with the bounce magic?
    (Lets a shared channel distinguish bounce control traffic.)"""
    return len(blob) >= 4 and _U32.unpack_from(blob, 0)[0] == BOUNCE_MAGIC


def decode_header(blob: bytes) -> Optional[BounceMsgHeader]:
    """Decode the fixed header. Returns ``None`` on a short blob, bad magic,
    version mismatch, or a payload length the blob cannot satisfy."""
    if len(blob) < _HEADER.size:
        return None
    header = BounceMsgHeader(*_HEADER.unpack_from(blob, 0))
    if header.magic != BOUNCE_MAGIC or header.version != BOUNCE_VERSION:
        return None
    if len(blob) < _HEADER.size + header.payload_bytes:
        return None
    return header


def decode_credits(blob: bytes, header: BounceMsgHeader) -> Optional[list[CreditEntry]]:
    """Decode GRANT credits (``header.count`` entries). ``None`` if malformed."""
    expect = header.count * _CREDIT.size
    if expect != header.payload_bytes or len(blob) < _HEADER.size + expect:
        return None
    return [
        CreditEntry(*fields)
        for fields in _CREDIT.iter_unpack(blob[_HEADER.size : _HEADER.size + expect])
    ]


def decode_scatter(blob: bytes, header: BounceMsgHeader) -> Optional[np.ndarray]:
    """Decode DATA scatter runs into an owning [count] structured array of
    ``SCATTER_RUN_DTYPE``. ``None`` if malformed."""
    expect = header.count * SCATTER_RUN_DTYPE.itemsize
    if expect != header.payload_bytes or len(blob) < _HEADER.size + expect:
        return None
    view = np.frombuffer(blob, dtype=SCATTER_RUN_DTYPE, count=header.count, offset=_HEADER.size)
    return view.copy()  # decouple from the (possibly reused) message buffer


def decode_want(blob: bytes, header: BounceMsgHeader) -> Optional[tuple[list[int], str]]:
    """Decode a WANT into (per-chunk byte sizes, sender endpoint). An empty
    size list means cancel (see :func:`is_cancel_want`). ``None`` if
    malformed."""
    sizes_bytes = header.count * 4
    off = _HEADER.size
    # Need the chunk sizes plus the u32 endpoint-length prefix.
    if len(blob) < off + sizes_bytes + 4:
        return None
    (ep_len,) = _U32.unpack_from(blob, off + sizes_bytes)
    expect_payload = sizes_bytes + 4 + ep_len
    if header.payload_bytes != expect_payload or len(blob) < off + expect_payload:
        return None
    sizes = np.frombuffer(blob, dtype=np.uint32, count=header.count, offset=off)
    ep_start = off + sizes_bytes + 4
    try:
        endpoint = blob[ep_start : ep_start + ep_len].decode("utf-8")
    except UnicodeDecodeError:
        return None  # endpoints are ascii zmq addresses; anything else is junk
    return sizes.tolist(), endpoint


def decode_ack(blob: bytes, header: BounceMsgHeader) -> Optional[list[AckEntry]]:
    """Decode a v3 batched ACK into ``(chunk_idx, region_handle)`` entries.
    ``None`` if malformed."""
    expect = header.count * _ACK_ENTRY.size
    if expect != header.payload_bytes or len(blob) < _HEADER.size + expect:
        return None
    return [
        AckEntry(*fields)
        for fields in _ACK_ENTRY.iter_unpack(blob[_HEADER.size : _HEADER.size + expect])
    ]


def is_cancel_want(chunk_sizes: Sequence[int]) -> bool:
    """Does a decoded WANT mean "cancel/retract"? (An empty chunk list.)"""
    return len(chunk_sizes) == 0
