#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send the launcher shutdown message without importing TensorRT-LLM or MPI."""

import ctypes
import hashlib
import hmac
import math
import os

# Only serialize fixed protocol values; never deserialize input.
import pickle  # nosec B403
import re
import sys
import time

import zmq

# Match the stop protocol in mgmn_leader_node without importing TensorRT-LLM.
STOP_ACK_TIMEOUT_SECONDS = 1.0
STOP_MAX_ATTEMPTS = 300
# ZeroMqQueue uses plain pickle.dumps. Match its fixed ACK encodings instead of
# unpickling worker messages, which could import TensorRT-LLM.
_ACK_PAYLOADS = frozenset(
    pickle.dumps("shutdown-ack", protocol=protocol)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
)


def _is_shutdown_ack(frame: bytes, hmac_key: bytes) -> bool:
    """Authenticate the fixed ACK without loading worker result/error objects."""
    signature_size = hashlib.sha256().digest_size
    if len(frame) <= signature_size:
        return False
    payload, signature = frame[:-signature_size], frame[-signature_size:]
    return (
        hmac.compare_digest(signature, hmac.digest(hmac_key, payload, "sha256"))
        and payload in _ACK_PAYLOADS
    )


def _consume_hmac_key() -> bytes:
    """Read the session key and erase its inherited environment value.

    Returns:
        The decoded 32-byte session key.

    Raises:
        KeyError: The launcher did not supply a session key.
        ValueError: The key is not exactly 64 hexadecimal characters.
    """
    # Keep validation and scrubbing aligned with executor/utils.py's
    # _normalize_spawn_proxy_process_ipc_hmac_key and _scrub_process_env_value.
    # Duplicate these small operations here to avoid importing TensorRT-LLM.
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


def main(
    ack_timeout: float = STOP_ACK_TIMEOUT_SECONDS, max_attempts: int = STOP_MAX_ATTEMPTS
) -> int:
    """Retry the launcher's authenticated stop request until it is acknowledged.

    A fresh connection retries requests dropped while the server still has its
    previous PAIR client. An ACK means the server accepted the request, not that
    its workers have finished shutting down.

    Args:
        ack_timeout: Seconds to wait for the ACK after each send.
        max_attempts: Maximum number of fresh connections to try.

    Returns:
        Zero after acknowledgment, or one after reporting invalid configuration,
        a ZeroMQ failure, or exhausted retries to stderr.
    """
    try:
        hmac_key = _consume_hmac_key()
        address = os.environ["TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"]
        # Match ZeroMqQueue._prepare_data: pickle followed by HMAC-SHA256.
        message = pickle.dumps(None)
        signature = hmac.new(hmac_key, message, hashlib.sha256).digest()
        with zmq.Context() as context:
            for _ in range(max_attempts):
                with context.socket(zmq.PAIR) as socket:
                    # Never wait on close for a request the peer did not accept.
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.setsockopt(zmq.SNDTIMEO, math.ceil(ack_timeout * 1000))
                    socket.connect(address)
                    try:
                        socket.send(message + signature)
                        deadline = time.monotonic() + ack_timeout
                        while (remaining := deadline - time.monotonic()) > 0:
                            if not socket.poll(math.ceil(remaining * 1000)):
                                break
                            if _is_shutdown_ack(socket.recv(), hmac_key):
                                return 0
                    except zmq.ZMQError:
                        # Retry transport failures on a fresh PAIR connection.
                        continue
    except (KeyError, ValueError, zmq.ZMQError) as error:
        sys.stderr.write(f"Failed to send MPI Comm server shutdown: {error}\n")
        return 1
    sys.stderr.write(
        "Failed to send MPI Comm server shutdown: "
        f"no acknowledgment after {max_attempts} attempts\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
