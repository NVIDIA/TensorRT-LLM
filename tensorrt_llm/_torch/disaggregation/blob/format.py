# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The object format: the payload bytes followed by a fixed-size trailer.

Private and versioned. The trailer lets a reader check that an object has the expected format,
belongs to the expected name and has the expected length; it is not a payload checksum and proves
neither a complete commit nor the end of access. Any incompatible change bumps ``FORMAT_VERSION``.
"""

from __future__ import annotations

import hashlib
import struct

FORMAT_VERSION = 1
MAGIC = b"TLBK"

# magic, version, 2 bytes of padding, name digest, payload length.
_TRAILER = struct.Struct("<4sH2x16sQ")
TRAILER_BYTES = _TRAILER.size
_DIGEST_BYTES = 16


def name_digest(name: bytes) -> bytes:
    """The 16-byte digest of an object name that the trailer carries."""
    return hashlib.blake2b(name, digest_size=_DIGEST_BYTES).digest()


def encode_trailer(name: bytes, payload_nbytes: int) -> bytes:
    """The trailer for an object of ``payload_nbytes`` bytes stored under ``name``."""
    if payload_nbytes <= 0:
        raise ValueError(f"payload length must be positive, got {payload_nbytes}")
    return _TRAILER.pack(MAGIC, FORMAT_VERSION, name_digest(name), payload_nbytes)


def check_trailer(trailer: bytes, name: bytes, payload_nbytes: int) -> str | None:
    """``None`` when ``trailer`` matches ``name`` and ``payload_nbytes``, else why it does not."""
    if len(trailer) != TRAILER_BYTES:
        return f"trailer is {len(trailer)} bytes, expected {TRAILER_BYTES}"
    magic, version, digest, length = _TRAILER.unpack(trailer)
    if magic != MAGIC:
        return "bad magic"
    if version != FORMAT_VERSION:
        return f"format version {version}, expected {FORMAT_VERSION}"
    if digest != name_digest(name):
        return "object name mismatch"
    if length != payload_nbytes:
        return f"payload length {length}, expected {payload_nbytes}"
    return None
