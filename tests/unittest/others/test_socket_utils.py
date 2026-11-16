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


class ScriptedSocket:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = iter(chunks)
        self.timeouts: list[float] = []
        self.recv_calls = 0

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def recv(self, size: int) -> bytes:
        self.recv_calls += 1
        return next(self.chunks)[:size]


def test_recv_exact_preserves_absolute_deadline_across_fragments(monkeypatch) -> None:
    monotonic_values = iter([10.0, 11.0, 13.0])
    monkeypatch.setattr("test_common.socket_utils.time.monotonic", lambda: next(monotonic_values))
    connection = ScriptedSocket([b"shut", b"down"])

    assert recv_exact(connection, 8, 5) == b"shutdown"
    assert connection.timeouts == [4.0, 2.0]


def test_recv_exact_times_out_before_receiving_after_deadline(monkeypatch) -> None:
    monotonic_values = iter([10.0, 15.0])
    monkeypatch.setattr("test_common.socket_utils.time.monotonic", lambda: next(monotonic_values))
    connection = ScriptedSocket([b"shutdown"])

    with pytest.raises(TimeoutError):
        recv_exact(connection, 8, 5)
    assert connection.timeouts == []
    assert connection.recv_calls == 0
