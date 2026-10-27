# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fixed-size trailer the in-memory driver keeps beside each object.

The driver boundary leaves object format and integrity to each driver. A real store checks sizes,
hides unfinished objects and can checksum natively; the in-memory driver stands in for that with
this trailer, which it verifies on every get before a row counts as READY: version, key digest,
payload length and, when an algorithm is set, a payload checksum.
"""

from __future__ import annotations

import enum
import hashlib
import struct
import zlib

TRAILER_VERSION = 1
MAGIC = b"TLBK"


class Checksum(enum.IntEnum):
    """Payload checksum algorithms a trailer can name."""

    NONE = 0
    CRC32 = 1


# magic, trailer version, checksum algorithm, name digest, payload length, payload checksum.
_TRAILER = struct.Struct("<4sHH16sQQ")
TRAILER_BYTES = _TRAILER.size
_DIGEST_BYTES = 16


def name_digest(name: bytes) -> bytes:
    """The 16-byte digest of an object name that the trailer carries."""
    return hashlib.blake2b(name, digest_size=_DIGEST_BYTES).digest()


def payload_checksum(algorithm: Checksum, payload: bytes | memoryview) -> int:
    """The checksum of ``payload`` under ``algorithm``; 0 for ``Checksum.NONE``."""
    if algorithm is Checksum.NONE:
        return 0
    return zlib.crc32(payload) & 0xFFFFFFFF


def encode_trailer(
    name: bytes,
    payload_nbytes: int,
    *,
    checksum: Checksum = Checksum.NONE,
    payload: bytes | memoryview | None = None,
) -> bytes:
    """The trailer for an object of ``payload_nbytes`` bytes stored under ``name``.

    Args:
        name: The object's name; the in-memory driver uses the store key.
        payload_nbytes: The payload length.
        checksum: The algorithm to checksum the payload with.
        payload: The payload bytes; required when ``checksum`` is not ``NONE``.
    """
    if payload_nbytes <= 0:
        raise ValueError(f"payload length must be positive, got {payload_nbytes}")
    value = 0
    if checksum is not Checksum.NONE:
        if payload is None or len(payload) != payload_nbytes:
            raise ValueError("a checksum needs the whole payload")
        value = payload_checksum(checksum, payload)
    return _TRAILER.pack(MAGIC, TRAILER_VERSION, checksum, name_digest(name), payload_nbytes, value)


def check_trailer(
    trailer: bytes, name: bytes, payload_nbytes: int, payload: bytes | memoryview | None = None
) -> str | None:
    """``None`` when ``trailer`` matches ``name``, ``payload_nbytes`` and, if it names an algorithm,
    the checksum of ``payload``; otherwise why it does not. Every trailer byte is checked."""
    if len(trailer) != TRAILER_BYTES:
        return f"trailer is {len(trailer)} bytes, expected {TRAILER_BYTES}"
    magic, version, algorithm, digest, length, value = _TRAILER.unpack(trailer)
    if magic != MAGIC:
        return "bad magic"
    if version != TRAILER_VERSION:
        return f"trailer version {version}, expected {TRAILER_VERSION}"
    try:
        checksum = Checksum(algorithm)
    except ValueError:
        return f"unknown checksum algorithm {algorithm}"
    if digest != name_digest(name):
        return "object name mismatch"
    if length != payload_nbytes:
        return f"payload length {length}, expected {payload_nbytes}"
    if checksum is Checksum.NONE:
        return None if value == 0 else "checksum set without an algorithm"
    if payload is None:
        return "checksum needs the payload"
    if value != payload_checksum(checksum, payload):
        return "payload checksum mismatch"
    return None
