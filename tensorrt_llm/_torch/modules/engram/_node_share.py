# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A node-local resource that the first process to claim its name loads and then serves, as file descriptors,
to the user's other processes, over an abstract UNIX socket."""

import contextlib
import errno
import functools
import json
import os
import socket
import struct
import threading
import time
import weakref

from tensorrt_llm.logger import logger

# A node-share name that stays bound this long without being served has no live owner.
_UNSERVED_TIMEOUT_S = 30
_READY = b"ready"


def _peer_uid(connection: socket.socket) -> int:
    credentials = connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
    )
    return struct.unpack("3i", credentials)[1]


class _Served:
    """What a node-share owner serves: descriptors and a payload, then a ready notice; joiners
    that arrive before the notice keep their connections open for it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.fds: list[int] = []
        self.payload = b""
        self.ready = False
        self.closed = False
        self.waiting: list[socket.socket] = []


def _serve(listener: socket.socket, served: _Served) -> None:
    """Send the descriptors and payload to each same-user peer until the listener closes."""
    try:
        while True:
            try:
                connection, _ = listener.accept()
            except OSError as error:
                if error.errno in (errno.EINVAL, errno.EBADF):
                    return
                logger.warning(f"Engram node share: accept failed: {error}")
                # Persistent errors such as EMFILE would otherwise spin.
                time.sleep(1)
                continue
            with served.lock:
                if served.closed:
                    connection.close()
                    return
                try:
                    if _peer_uid(connection) == os.getuid():
                        socket.send_fds(connection, [served.payload], served.fds)
                        if not served.ready:
                            served.waiting.append(connection)
                            continue
                        connection.send(_READY)
                except OSError as error:
                    logger.warning(f"Engram node share: could not serve a peer: {error}")
                connection.close()
    finally:
        # A name that no thread serves must not stay bound; its waiting joiners elect again.
        with contextlib.suppress(OSError):
            listener.shutdown(socket.SHUT_RDWR)
        listener.close()
        with served.lock:
            for connection in served.waiting:
                connection.close()
            served.waiting.clear()


def _close_listener(listener: socket.socket, served: _Served) -> None:
    # Shutdown wakes a blocked accept and refuses new peers; close frees the name.
    with contextlib.suppress(OSError):
        listener.shutdown(socket.SHUT_RDWR)
    listener.close()
    with served.lock:
        served.closed = True
        for connection in served.waiting:
            connection.close()
        served.waiting.clear()
        for fd in served.fds:
            os.close(fd)
        served.fds = []


def _forget_in_child(listener: weakref.ReferenceType, served: _Served) -> None:
    # A forked child neither serves nor keeps the name, the joiners' connections or the memory alive;
    # it closes only its own copies (no shutdown), and the serving thread may have held the lock.
    served.lock = threading.Lock()
    child_listener = listener()
    if child_listener is not None:
        child_listener.close()
    for connection in served.waiting:
        connection.close()
    served.waiting = []
    for fd in served.fds:
        os.close(fd)
    served.fds = []
    served.closed = True


class NodeShare:
    """A resource of this node that the first process to claim its name allocates, serves as file
    descriptors to the user's other processes, fills, and marks ready; it stops when closed,
    collected, or at exit."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._listener: socket.socket | None = None
        self._served = _Served()
        self._finalizer: weakref.finalize | None = None
        # A joiner's connection to its owner, open until the owner's ready notice or exit.
        self._connection: socket.socket | None = None

    def join(self, claim_after_s: float = 0.0) -> tuple[list[int], dict] | None:
        """Claim the name (None), not before ``claim_after_s``, or wait for its owner's descriptors and
        metadata, claiming it again if that owner goes away first; the caller closes the descriptors
        and then calls ``wait_ready``, which the owner answers only once the content is complete."""
        claim_at = time.monotonic() + claim_after_s
        deadline = claim_at + _UNSERVED_TIMEOUT_S
        while True:
            if time.monotonic() >= claim_at and self._claim():
                return None
            client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                try:
                    client.connect("\0" + self.name)
                except ConnectionRefusedError:
                    # Unclaimed yet, or bound but not listening: an owner starting up or shutting down.
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"node share {self.name} is held but never served")
                    time.sleep(0.1)
                    continue
                if _peer_uid(client) != os.getuid():
                    raise PermissionError(f"node share {self.name} is held by another user")
                try:
                    payload, fds, _, _ = socket.recv_fds(client, 1 << 16, 1)
                except ConnectionResetError:
                    payload, fds = b"", []
                if payload:
                    try:
                        meta = json.loads(payload)
                    except ValueError:
                        for fd in fds:
                            os.close(fd)
                        raise
                    self._connection, client = client, None
                    return fds, meta
            finally:
                if client is not None:
                    client.close()
            logger.info(f"Node share {self.name}: its owner left before publishing; electing again")
            deadline = time.monotonic() + _UNSERVED_TIMEOUT_S

    def wait_ready(self) -> bool:
        """Joiner: block until the owner marks the content ready (True) or goes away first (False)."""
        connection, self._connection = self._connection, None
        if connection is None:
            return False
        with connection:
            try:
                return connection.recv(len(_READY)) == _READY
            except OSError:
                return False

    def _claim(self) -> bool:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            listener.bind("\0" + self.name)
            listener.listen(64)
        except OSError as error:
            listener.close()
            if error.errno == errno.EADDRINUSE:
                return False
            raise
        self._listener = listener
        self._finalizer = weakref.finalize(self, _close_listener, listener, self._served)
        os.register_at_fork(
            after_in_child=functools.partial(_forget_in_child, weakref.ref(listener), self._served)
        )
        return True

    def publish(self, fds: list[int], meta: dict) -> None:
        """Owner: serve ``fds``, which this share then owns, and ``meta`` to joiners, who wait for
        ``mark_ready`` before they use the content."""
        with self._served.lock:
            self._served.fds = fds
            self._served.payload = json.dumps(meta).encode()
        threading.Thread(
            target=_serve,
            args=(self._listener, self._served),
            name="engram-node-share",
            daemon=True,
        ).start()

    def mark_ready(self) -> None:
        """Owner: the content is complete; release the joiners waiting for it."""
        with self._served.lock:
            self._served.ready = True
            for connection in self._served.waiting:
                with contextlib.suppress(OSError):
                    connection.send(_READY)
                connection.close()
            self._served.waiting.clear()

    def close(self) -> None:
        """Owner: stop serving and free the name, so waiting joiners elect another owner. Joiner:
        drop the connection to the owner."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._finalizer is not None:
            self._finalizer()
