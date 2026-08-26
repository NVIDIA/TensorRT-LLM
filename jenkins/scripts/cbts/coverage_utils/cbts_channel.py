# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Broadcast of the current test context from the outer pytest to its subprocesses.

One producer (the pytest process, via ``cbts_plugin``) owns the context and announces
every change over an ``AF_UNIX`` socket. Consumers (``sitecustomize`` in product
processes) subscribe and are handed the context that is current at that instant, then
every later announcement, until the producer's final ``STOP``.

Announcing and (un)subscribing are mutually exclusive, so a process that joins while an
announcement is in flight sees either the old or the new context, never neither: a
subscriber is registered and sent the current context under one lock.

Consumers acknowledge each announcement, which lets the producer wait until every live
consumer has switched before a test body starts, rather than leaving a window in which a
reused pool worker still answers to the previous test. ``STOP`` then replaces a
stop-file: once the producer has seen every consumer unsubscribe, their coverage is
known to be on disk.

Wire format is one newline-terminated frame per message:
``C <seq> <nodeid>`` (context), ``X <seq>`` (stop), ``A <seq>`` (acknowledgement).
"""

import errno
import os
import selectors
import socket
import tempfile
import threading
import time

ADDRESS_ENV = "CBTS_CONTEXT_SOCKET"

_CONTEXT = "C"
_STOP = "X"
_ACK = "A"

# AF_UNIX paths are capped near 108 bytes, well below what a CI workspace path can reach,
# so the socket lives in a short per-user directory rather than beside the coverage data.
_SOCKET_DIR = os.path.join(tempfile.gettempdir(), f"cbts-{os.getuid()}")

_DEFAULT_ACK_TIMEOUT = 5.0
_DEFAULT_DRAIN_TIMEOUT = 30.0
_DEFAULT_JOIN_TIMEOUT = 2.0
_SEND_TIMEOUT = 2.0


def default_address():
    """A fresh address for a producer in this process."""
    return os.path.join(_SOCKET_DIR, f"ctx-{os.getpid()}.sock")


def announced_address():
    """The address a consumer was told to use, or "" if nobody told it one.

    Reaches an ordinary child through the environment it was exec'd with, and an
    ``mpi4py`` pool worker through the ``env`` payload the patched ``MPIPoolExecutor``
    forwards (applied during the worker's sync handshake, before it runs any task).
    """
    return os.environ.get(ADDRESS_ENV, "").strip()


def _send(conn, text):
    """Best-effort frame write; False means the peer is gone."""
    try:
        conn.sendall((text + "\n").encode("utf-8"))
        return True
    except OSError:
        return False


def _frames(buffer, chunk):
    """Split accumulated bytes into complete frames, returning (remainder, frames)."""
    buffer += chunk
    *complete, remainder = buffer.split(b"\n")
    return remainder, [line.decode("utf-8", "replace") for line in complete if line]


class ContextServer:
    """Producer: owns the current context, broadcasts changes, collects acknowledgements."""

    def __init__(self, address):
        self.address = address
        self._lock = threading.Lock()
        # Signalled whenever an acknowledgement lands or a subscriber leaves.
        self._progress = threading.Condition(self._lock)
        self._acked = {}  # conn -> highest sequence acknowledged
        self._buffers = {}  # conn -> bytes not yet forming a frame
        self._context = ""
        self._sequence = 0
        self._stopped = False
        self._listener = None
        self._wake_r = None
        self._wake_w = None
        self._thread = None

    @classmethod
    def start(cls, address=None):
        """Bind and serve, or return None if the socket cannot be created."""
        address = address or announced_address() or default_address()
        server = cls(address)
        try:
            os.makedirs(os.path.dirname(address), mode=0o700, exist_ok=True)
            try:
                os.unlink(address)
            except FileNotFoundError:
                pass
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(address)
            listener.listen(128)
        except OSError:
            return None
        server._listener = listener
        server._wake_r, server._wake_w = socket.socketpair()
        server._thread = threading.Thread(
            target=server._serve, daemon=True, name="cbts-context-server"
        )
        server._thread.start()
        return server

    def _serve(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self._listener, selectors.EVENT_READ)
            selector.register(self._wake_r, selectors.EVENT_READ)
            while True:
                for key, _ in selector.select():
                    if key.fileobj is self._wake_r:
                        try:
                            self._wake_r.recv(4096)
                        except OSError:
                            pass
                        return
                    if key.fileobj is self._listener:
                        self._accept(selector)
                    else:
                        self._read_ack(selector, key.fileobj)

    def _accept(self, selector):
        try:
            conn, _ = self._listener.accept()
        except OSError:
            return
        conn.settimeout(_SEND_TIMEOUT)
        with self._lock:
            if self._stopped:
                # Subscribing after the final announcement is refused, so a late process
                # never waits for a context that will not arrive.
                _send(conn, f"{_STOP} {self._sequence}")
                conn.close()
                return
            self._acked[conn] = 0
            self._buffers[conn] = b""
            delivered = _send(conn, f"{_CONTEXT} {self._sequence} {self._context}")
            if not delivered:
                self._drop_locked(conn)
                conn.close()
                return
        selector.register(conn, selectors.EVENT_READ)

    def _read_ack(self, selector, conn):
        try:
            chunk = conn.recv(4096)
        except OSError:
            chunk = b""
        if not chunk:
            selector.unregister(conn)
            with self._progress:
                self._drop_locked(conn)
                self._progress.notify_all()
            conn.close()
            return
        with self._progress:
            remainder, frames = _frames(self._buffers.get(conn, b""), chunk)
            self._buffers[conn] = remainder
            for frame in frames:
                kind, _, payload = frame.partition(" ")
                if kind != _ACK:
                    continue
                try:
                    sequence = int(payload)
                except ValueError:
                    continue
                if sequence > self._acked.get(conn, 0):
                    self._acked[conn] = sequence
            self._progress.notify_all()

    def _drop_locked(self, conn):
        self._acked.pop(conn, None)
        self._buffers.pop(conn, None)

    def announce(self, nodeid, ack_timeout=_DEFAULT_ACK_TIMEOUT):
        """Publish a new context and, by default, wait for every consumer to confirm it."""
        with self._lock:
            if self._stopped:
                return True
            self._sequence += 1
            sequence = self._sequence
            self._context = nodeid
            for conn in list(self._acked):
                if not _send(conn, f"{_CONTEXT} {sequence} {nodeid}"):
                    self._drop_locked(conn)
        if not ack_timeout:
            return True
        return self.wait_acks(sequence, ack_timeout)

    def wait_acks(self, sequence, timeout=_DEFAULT_ACK_TIMEOUT):
        """True once every live consumer has acknowledged ``sequence``.

        A consumer that misses the deadline is ejected rather than waited on again: it is
        not tracking the context anyway, and leaving it registered would charge every later
        announcement the full timeout. Ejection closes its socket, so it saves what it has
        and stops recording instead of attributing later work to a stale test.
        """
        deadline = time.monotonic() + timeout
        with self._progress:
            while True:
                pending = [conn for conn, acked in self._acked.items() if acked < sequence]
                if not pending:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    for conn in pending:
                        self._drop_locked(conn)
                        try:
                            conn.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return False
                self._progress.wait(remaining)

    @property
    def subscriber_count(self):
        with self._lock:
            return len(self._acked)

    def close(self, drain_timeout=_DEFAULT_DRAIN_TIMEOUT):
        """Announce the final message, then wait for every consumer to unsubscribe."""
        with self._lock:
            if self._listener is None:
                return True
            if not self._stopped:
                self._stopped = True
                self._sequence += 1
                for conn in list(self._acked):
                    if not _send(conn, f"{_STOP} {self._sequence}"):
                        self._drop_locked(conn)
        deadline = time.monotonic() + drain_timeout
        with self._progress:
            while self._acked:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._progress.wait(remaining)
            drained = not self._acked
        if self._wake_w is not None:
            try:
                self._wake_w.send(b"x")
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(_DEFAULT_JOIN_TIMEOUT)
        for sock in (self._listener, self._wake_r, self._wake_w):
            try:
                sock.close()
            except OSError:
                pass
        self._listener = None
        try:
            os.unlink(self.address)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                pass
        return drained


class ContextSubscriber:
    """Consumer: applies the current context, then every announcement until STOP."""

    def __init__(self, conn, on_context, on_stop):
        self._conn = conn
        self._on_context = on_context
        self._on_stop = on_stop
        self._buffer = b""
        self._thread = None

    @classmethod
    def subscribe(cls, address=None, *, on_context, on_stop, timeout=2.0):
        """Join the channel and apply the current context, or return None.

        The first frame is read here rather than on the reader thread, so a caller that
        returns from ``subscribe`` is already on the right context.
        """
        address = address or announced_address()
        if not address:
            return None
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(timeout)
        try:
            conn.connect(address)
        except OSError:
            conn.close()
            return None
        subscriber = cls(conn, on_context, on_stop)
        if not subscriber._pump(block=True):
            conn.close()
            return None
        conn.settimeout(None)
        subscriber._thread = threading.Thread(
            target=subscriber._run, daemon=True, name="cbts-context-subscriber"
        )
        subscriber._thread.start()
        return subscriber

    def _pump(self, block=False):
        """Read one chunk and dispatch its frames; False once the channel is finished."""
        try:
            chunk = self._conn.recv(4096)
        except OSError:
            return False
        if not chunk:
            # The producer exited without a STOP; treat it as one so coverage is saved.
            self._on_stop()
            return False
        self._buffer, frames = _frames(self._buffer, chunk)
        for frame in frames:
            kind, _, payload = frame.partition(" ")
            sequence, _, nodeid = payload.partition(" ")
            if kind == _CONTEXT:
                self._on_context(nodeid)
            elif kind == _STOP:
                self._on_stop()
                _send(self._conn, f"{_ACK} {sequence}")
                return False
            else:
                continue
            _send(self._conn, f"{_ACK} {sequence}")
        del block
        return True

    def _run(self):
        while self._pump():
            pass
        self.close()

    def close(self):
        """Leave the channel, waking our own reader and signalling EOF to the producer."""
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._conn.close()
        except OSError:
            pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(_DEFAULT_JOIN_TIMEOUT)
