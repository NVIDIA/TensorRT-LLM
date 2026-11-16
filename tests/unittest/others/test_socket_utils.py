# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket

import pytest
from test_common.socket_utils import recv_exact

pytestmark = pytest.mark.cpu_only


class FragmentedSocket:
    """Limit each read so a complete message requires multiple recv calls."""

    def __init__(self, connection: socket.socket, fragment_size: int) -> None:
        self.connection = connection
        self.fragment_size = fragment_size

    def settimeout(self, timeout: float) -> None:
        self.connection.settimeout(timeout)

    def recv(self, size: int) -> bytes:
        return self.connection.recv(min(size, self.fragment_size))


def test_recv_exact_accumulates_fragmented_message() -> None:
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(b"shutdown")
        assert recv_exact(FragmentedSocket(receiver, 2), 8, 1) == b"shutdown"
    finally:
        receiver.close()
        sender.close()


def test_recv_exact_rejects_eof_before_complete_message() -> None:
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(b"shut")
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError) as error:
            recv_exact(receiver, 8, 1)
        assert error.value.args == (b"shut",)
    finally:
        receiver.close()
        sender.close()
