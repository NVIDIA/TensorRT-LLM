# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HMAC-SHA256 framing of serialized IPC payloads.

A frame is the payload followed by its HMAC-SHA256 digest. This module uses
only the standard library, so the process-watch helper can share it with
IpcQueue without importing TensorRT-LLM.
"""

import hashlib
import hmac

DIGEST_SIZE = hashlib.sha256().digest_size


class HmacVerificationError(RuntimeError):
    """A received frame does not carry a valid HMAC for the shared key."""


def sign_frame(key: bytes, payload: bytes) -> bytes:
    """Return payload followed by its HMAC-SHA256 digest under key."""
    return payload + hmac.digest(key, payload, hashlib.sha256)


def verify_frame(key: bytes, frame: bytes) -> bytes:
    """Return the payload of a frame signed with key.

    Raises:
        HmacVerificationError: If the digest does not match, including for a
            frame shorter than the digest.
    """
    payload, digest = frame[:-DIGEST_SIZE], frame[-DIGEST_SIZE:]
    if not hmac.compare_digest(hmac.digest(key, payload, hashlib.sha256), digest):
        raise HmacVerificationError("HMAC verification failed")
    return payload
