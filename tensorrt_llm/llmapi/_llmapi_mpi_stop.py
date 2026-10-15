#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send the launcher shutdown message without importing TensorRT-LLM or MPI."""

import ctypes
import hashlib
import hmac
import os

# Only serialize the fixed shutdown sentinel; never deserialize input.
import pickle  # nosec B403
import re
import sys

import zmq


def _consume_hmac_key() -> bytes:
    """Read the session key and erase its inherited environment value.

    Returns:
        The decoded 32-byte session key.

    Raises:
        KeyError: The launcher did not supply a session key.
        ValueError: The key is not exactly 64 hexadecimal characters.
    """
    name = "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"
    key = os.environ[name]
    if sys.platform == "linux":
        # Removing os.environ alone leaves the original bytes in /proc/environ.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.getenv.restype = ctypes.c_void_p
        value_ptr = libc.getenv(os.fsencode(name))
        if value_ptr:
            ctypes.memset(value_ptr, 0, len(os.fsencode(key)))
    os.environ.pop(name, None)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", key):
        raise ValueError("IPC HMAC key must be a 64-character hex string.")
    return bytes.fromhex(key)


def main() -> int:
    """Send an authenticated stop request using the launcher's environment.

    The peer must connect within five seconds. Socket cleanup has a one-second
    linger limit if the peer disappears after the send.

    Returns:
        Zero after sending, or one after reporting invalid configuration or a
        ZeroMQ failure to stderr.
    """
    try:
        hmac_key = _consume_hmac_key()
        address = os.environ["TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"]
        # Match ZeroMqQueue._prepare_data: pickle followed by HMAC-SHA256.
        message = pickle.dumps(None)
        signature = hmac.new(hmac_key, message, hashlib.sha256).digest()
        with zmq.Context() as context, context.socket(zmq.PAIR) as socket:
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.setsockopt(zmq.SNDTIMEO, 5000)
            socket.setsockopt(zmq.LINGER, 1000)
            socket.connect(address)
            socket.send(message + signature)
    except (KeyError, ValueError, zmq.ZMQError) as error:
        sys.stderr.write(f"Failed to send MPI Comm server shutdown: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
