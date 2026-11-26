# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
import time


def recv_exact(connection: socket.socket, size: int, timeout: float) -> bytes:
    """Receive exactly ``size`` bytes under one absolute timeout."""
    deadline = time.monotonic() + timeout
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        connection.settimeout(remaining)
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError(bytes(chunks))
        chunks.extend(chunk)
    return bytes(chunks)
