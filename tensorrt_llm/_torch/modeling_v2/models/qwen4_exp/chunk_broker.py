# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand one process's exportable allocation to the other processes of the host that ask for it.

A segment is named by a key. The first process to bind the key's abstract Unix socket
(`\\0trtllm-ple-<uid>-<key>`) owns it: it creates and fills the allocation, then serves the
allocation's file descriptor to every process that connects (`SCM_RIGHTS`; a peer of another
uid is refused through `SO_PEERCRED`). A process that cannot bind connects and waits. When the
owner exits or aborts before publishing, the kernel closes its sockets and the waiter tries to
own the segment itself. Nothing is written to the filesystem, so a crash leaves nothing behind.

Abstract socket names belong to a network namespace: processes in different namespaces do not
meet, and each keeps its own copy.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import struct
import threading
import time
from typing import Callable, Optional

OWNER = "owner"
ATTACHED = "attached"

_CRED = struct.Struct("3i")  # struct ucred: pid, uid, gid
_POLL_SECONDS = 1.0


def socket_name(key: str) -> str:
    """The abstract socket name of `key` (per uid)."""
    return f"\0trtllm-ple-{os.getuid()}-{key}"


class OwnedSegment:
    """The owner's side: serves `publish`ed `(fd, meta)` to every same-uid peer, until `close`."""

    def __init__(self, server: socket.socket, key: str) -> None:
        self.key = key
        self._server = server
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._fd = -1
        self._meta: dict = {}
        self._answers: list[threading.Thread] = []
        self._thread = threading.Thread(
            target=self._serve, name=f"ple-broker-{key[:12]}", daemon=True
        )
        self._thread.start()

    def publish(self, fd: int, meta: dict) -> None:
        """Make the allocation available. `fd` stays owned by the caller and must stay open."""
        self._fd = fd
        self._meta = dict(meta)
        self._ready.set()

    def close(self) -> None:
        """Stop serving; peers still waiting see the connection close and retry."""
        self._closed.set()
        try:
            # Closing alone does not wake the thread blocked in `accept`; shutting down does.
            self._server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._server.close()
        for thread in [self._thread, *self._answers]:
            thread.join(_POLL_SECONDS * 5)

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            answer = threading.Thread(target=self._answer, args=(conn,), daemon=True)
            self._answers = [t for t in self._answers if t.is_alive()] + [answer]
            answer.start()

    def _answer(self, conn: socket.socket) -> None:
        with conn:
            try:
                _, uid, _ = _CRED.unpack(
                    conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _CRED.size)
                )
                if uid != os.getuid():
                    return
                while not self._ready.wait(_POLL_SECONDS):
                    if self._closed.is_set():
                        return
                if self._closed.is_set():
                    return
                socket.send_fds(conn, [json.dumps(self._meta).encode()], [self._fd])
            except OSError:
                return


def try_own(key: str) -> Optional[OwnedSegment]:
    """Own `key` if no process does (non-blocking), else None."""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    try:
        server.bind(socket_name(key))
        server.listen(64)
    except OSError as error:
        server.close()
        if error.errno == errno.EADDRINUSE:
            return None
        raise
    return OwnedSegment(server, key)


def acquire(
    key: str, timeout_s: float, log: Callable[[str], None] = print, report_s: float = 60.0
) -> tuple:
    """`(OWNER, OwnedSegment)` when no process owns `key`, else `(ATTACHED, fd, meta)` once
    the owner publishes; an owner that exits or aborts first hands `key` to its waiters.
    Raises `TimeoutError` when an owner has not published within `timeout_s`."""
    name = socket_name(key)
    deadline = time.monotonic() + timeout_s
    next_report = time.monotonic() + report_s
    while True:
        owned = try_own(key)
        if owned is not None:
            return OWNER, owned
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        try:
            conn.connect(name)
        except (ConnectionRefusedError, FileNotFoundError):
            # The owner just exited, or is between `bind` and `listen`.
            conn.close()
            time.sleep(0.05)
            continue
        got: Optional[tuple] = None
        try:
            while got is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no allocation for {key!r} within {timeout_s:.0f} s")
                conn.settimeout(min(remaining, report_s))
                try:
                    got = socket.recv_fds(conn, 4096, 1)
                except socket.timeout:
                    if time.monotonic() >= next_report:
                        log(f"ple host table: waiting for the owner of {key!r} to publish it")
                        next_report = time.monotonic() + report_s
        finally:
            conn.close()
        msg, fds, _, _ = got
        if fds:
            return ATTACHED, fds[0], json.loads(msg.decode())
        # The connection closed without a descriptor: the owner exited or aborted. Retry.
