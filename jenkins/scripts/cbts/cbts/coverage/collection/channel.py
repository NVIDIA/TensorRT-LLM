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
``C <seq> <nodeid>`` (context)
``X <seq>`` (stop)
``A <seq>`` (acknowledgement)
``L`` (a subscriber leaving on its own, before any ``STOP``)
``T <path>`` (a ticket file to hold for the subscriber's lifetime, see ``ContextServer._confirmed_dead``).
"""

from __future__ import annotations

import errno
import fcntl
import os
import selectors
import shutil
import socket
import sys
import tempfile
import threading
import time
from typing import Callable, Optional

ADDRESS_ENV = "CBTS_CONTEXT_SOCKET"

# (process_uid, test, kind, reason).
Taint = tuple[str, str, str, str]


class ChannelError(RuntimeError):
    """The context channel could not be opened or used as intended."""


# A peer disappearing is ordinary: a worker exits, a pool is torn down, a process is
# killed. These errnos say exactly that and carry nothing worth reporting; anything
# else is unexpected and its detail is worth the operator's attention.
_PEER_GONE = frozenset(
    (
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ENOTCONN,
        errno.ESHUTDOWN,
        errno.EBADF,
    )
)


def _report_unexpected(what: str, exc: OSError) -> None:
    """Surface an errno the protocol does not treat as routine."""
    if getattr(exc, "errno", None) not in _PEER_GONE:
        print(f"[cbts] {what}: {exc!r}", file=sys.stderr)


_CONTEXT = "C"
_STOP = "X"
_ACK = "A"
_IDENTITY = "I"
_LEAVE = "L"
_TICKET = "T"

# What is wrong with the data, stored per (process, test) alongside the reason. One
# event usually produces both, against different tests: when a subscriber falls behind
# on test A while test B runs, A gets ATTRIBUTION and B gets INCOMPLETE.
TAINT_ATTRIBUTION = "attribution"  # rows here may belong to another test
TAINT_INCOMPLETE = "incomplete"  # rows here are right, but some are missing

# Why a subscriber's coverage could not be vouched for, stored per (process, test).
# A taint whose test is the empty context is stage-scoped: it applies to every test,
# used when the recorder cannot say which tests it covers.

# -- Scoped: the reason names exactly the one test in doubt, no fan-out. --

TAINT_UNACKNOWLEDGED = "context_not_acknowledged"
# A subscriber whose EOF we can trace to a confirmed-dead process (see
# ``ContextServer._confirmed_dead``): it cannot have kept running past the test it was
# last on, so only that one test -- never announced to it as ended -- is in doubt.
TAINT_SUBSCRIBER_DEAD = "subscriber_confirmed_dead"
TAINT_NOT_DRAINED = "did_not_finish_before_deadline"
# The connection itself never delivered a context at accept time. This taints the
# current test directly; the subscriber is also dropped as unreachable, which taints
# every later test too -- see ``TAINT_STOPPED_RECORDING`` below.
TAINT_UNREACHABLE = "unreachable_on_subscribe"
TAINT_NO_CHANNEL = "no_context_channel"

# -- Fan-out: the reason cannot name which later tests are affected, so it taints --
# -- every test announced from the loss onward (see ``ContextServer.taints``). --

# A subscriber is ejected -- for failing to ACK in time, for never receiving a context at
# accept time, or for an EOF that is NOT a confirmed kill (still possibly running, e.g. a
# long-lived serving process or disagg helper stuck on the last context it saw) -- so
# every test announced from that point on is missing its coverage, or may have been
# misattributed to the stale context it never learned changed.
TAINT_STOPPED_RECORDING = "subscriber_stopped_recording"
# Recorded by the subscriber itself: a process that never joined is invisible to the
# producer, so nothing else can flag it. Stage-scoped of necessity -- it keeps recording
# under the context it was spawned with, so that context collects any later test's work
# while those tests lose its coverage entirely, and never having heard an announcement is
# exactly what stops it naming them.
TAINT_NOT_SUBSCRIBED = "context_channel_unreachable"
# Recorded by an MPI pool worker's own bootstrap (sitecustomize.py) when the backstop
# timer (CBTS_WORKER_ACTIVATE_MAX_SECONDS) fires before the framework import completes.
# An MPI pool worker always leaves its import phase unrecorded, by design, and that gap
# alone is not tainted. This reason is for the rarer case where the deferral itself ran
# unusually long: the default budget is already past the documented worst-case cold
# import, so hitting it signals an abnormally slow run worth flagging. Stage-scoped: the
# process cannot name which tests ran during the delay, only that it does not know.
TAINT_ACTIVATION_TIMEOUT = "worker_activation_timeout"

# AF_UNIX paths are capped near 108 bytes, well below what a CI workspace path can reach,
# so the socket lives in a short per-user directory rather than beside the coverage data.
_SOCKET_DIR = os.path.join(tempfile.gettempdir(), f"cbts-{os.getuid()}")

_DEFAULT_ACK_TIMEOUT = 5.0
_DEFAULT_DRAIN_TIMEOUT = 30.0
_DEFAULT_JOIN_TIMEOUT = 2.0
_SEND_TIMEOUT = 2.0


def default_address() -> str:
    """A fresh address for a producer in this process."""
    return os.path.join(_SOCKET_DIR, f"ctx-{os.getpid()}.sock")


def announced_address() -> str:
    """The address a consumer was told to use, or "" if nobody told it one.

    Reaches an ordinary child through the environment it was exec'd with, and an
    ``mpi4py`` pool worker through the ``env`` payload the patched ``MPIPoolExecutor``
    forwards (applied during the worker's sync handshake, before it runs any task).
    """
    return os.environ.get(ADDRESS_ENV, "").strip()


def _send(conn: socket.socket, text: str) -> bool:
    """Best-effort frame write; False means the peer is gone."""
    try:
        conn.sendall((text + "\n").encode("utf-8"))
        return True
    except OSError as exc:
        _report_unexpected("could not send on the context channel", exc)
        return False


def _frames(buffer: bytes, chunk: bytes) -> tuple[bytes, list[str]]:
    """Split accumulated bytes into complete frames, returning (remainder, frames)."""
    buffer += chunk
    *complete, remainder = buffer.split(b"\n")
    return remainder, [line.decode("utf-8", "replace") for line in complete if line]


class ContextServer:
    """Producer: owns the current context, broadcasts changes, collects acknowledgements."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._lock = threading.Lock()
        # Signalled whenever an acknowledgement lands or a subscriber leaves.
        self._progress = threading.Condition(self._lock)
        self._acked: dict[socket.socket, int] = {}  # conn -> highest sequence acknowledged
        self._buffers: dict[socket.socket, bytes] = {}  # conn -> bytes not yet forming a frame
        self._identity: dict[socket.socket, str] = {}  # conn -> subscriber-reported process uid
        self._left: set[socket.socket] = set()  # conns that sent LEAVE before disconnecting
        self._tickets: dict[socket.socket, str] = {}  # conn -> its assigned ticket file path
        self._next_ticket = 0
        self._ticket_dir = f"{address}.tickets"
        self._taints: set[Taint] = set()  # known outright
        self._announced: list[tuple[int, str]] = []  # (sequence, context) in announcement order
        # (process_uid, superseded_context or None, sequence): a subscriber that stopped
        # receiving announcements. Expanded when the taints are read.
        self._lost: list[tuple[str, Optional[str], int]] = []
        self._context = ""
        self._sequence = 0
        self._stopped = False
        self._listener: Optional[socket.socket] = None
        self._wake_r: Optional[socket.socket] = None
        self._wake_w: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def start(cls, address: Optional[str] = None) -> "ContextServer":
        """Bind and serve, raising ChannelError if the socket cannot be created."""
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
            shutil.rmtree(server._ticket_dir, ignore_errors=True)
            os.makedirs(server._ticket_dir, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ChannelError(f"could not bind the context channel at {address}") from exc
        server._listener = listener
        server._wake_r, server._wake_w = socket.socketpair()
        server._thread = threading.Thread(
            target=server._serve, daemon=True, name="cbts-context-server"
        )
        server._thread.start()
        return server

    def _serve(self) -> None:
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

    def _accept(self, selector: selectors.BaseSelector) -> None:
        try:
            conn, _ = self._listener.accept()
        except OSError as exc:
            _report_unexpected("could not accept a context subscriber", exc)
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
            self._next_ticket += 1
            ticket_path = os.path.join(self._ticket_dir, str(self._next_ticket))
            self._tickets[conn] = ticket_path
            # One write, so the caller's single blocking read at subscribe time is
            # guaranteed to see both frames together.
            delivered = _send(
                conn, f"{_TICKET} {ticket_path}\n{_CONTEXT} {self._sequence} {self._context}"
            )
            if not delivered:
                self._taint_locked(conn, TAINT_INCOMPLETE, TAINT_UNREACHABLE)
                # It never received a context and is dropped here, so like an ejected
                # subscriber it contributes nothing from now on -- recorded via the same
                # fan-out as TAINT_STOPPED_RECORDING (see the ``taints`` property). No
                # superseded context: whatever it was recording under, we never told it.
                self._lost.append((self._identity.get(conn, ""), None, self._sequence))
                self._drop_locked(conn)
                conn.close()
                return
        selector.register(conn, selectors.EVENT_READ)

    def _read_ack(self, selector: selectors.BaseSelector, conn: socket.socket) -> None:
        try:
            chunk = conn.recv(4096)
        except OSError as exc:
            _report_unexpected("could not read from a context subscriber", exc)
            chunk = b""
        if not chunk:
            selector.unregister(conn)
            with self._progress:
                # A client must acknowledge us if it's going to leave on its own.
                # An unnotified EOF can have 2 reasons:
                # 1. The client crashed. It cannot taint future successful tests by
                #    misattributing coverage.
                # 2. The client lost the channel connection, but is otherwise still
                #    functioning. It will misattribute coverage of future tests to
                #    the last test it acknowledged.
                # We can tell definitely which is the case by testing whether the
                # process is still alive - probing if the assigned file ticket
                # is still being held.
                if not self._stopped and conn in self._acked and conn not in self._left:
                    if self._confirmed_dead(conn):
                        # It cannot possibly still be running, so it cannot misattribute or
                        # miss work for a test it was never even told about: only the one it
                        # was on is in doubt.
                        self._taint_locked(conn, TAINT_INCOMPLETE, TAINT_SUBSCRIBER_DEAD)
                    else:
                        # Not confirmed dead: it may still be alive and running -- a long-lived
                        # subscriber (a serving process, a disagg helper) that keeps entering
                        # code now with no way to learn which test that belongs to.
                        # Same as missing ACK: every test announced from here on may go uncounted,
                        # or land on the stale context it's stuck on. They all can't be trusted.
                        self._lost.append(
                            (self._identity.get(conn, ""), self._context, self._sequence)
                        )
                self._drop_locked(conn)
                self._progress.notify_all()
            conn.close()
            return
        with self._progress:
            remainder, frames = _frames(self._buffers.get(conn, b""), chunk)
            self._buffers[conn] = remainder
            for frame in frames:
                kind, _, payload = frame.partition(" ")
                if kind == _IDENTITY:
                    # Names the leaf database this subscriber will write, so a taint
                    # can be attributed to the same process the coverage came from.
                    self._identity[conn] = payload
                    continue
                if kind == _LEAVE:
                    # This subscriber is unsubscribing on its own initiative, with its
                    # own final save already done. Nothing goes wrong.
                    self._left.add(conn)
                    continue
                if kind != _ACK:
                    continue
                try:
                    sequence = int(payload)
                except ValueError:
                    continue
                if sequence > self._acked.get(conn, 0):
                    self._acked[conn] = sequence
            self._progress.notify_all()

    def _taint_locked(
        self, conn: socket.socket, kind: str, reason: str, context: Optional[str] = None
    ) -> None:
        """Note that this subscriber's coverage for ``context`` is not trustworthy."""
        if context is None:
            context = self._context
        self._taints.add((self._identity.get(conn, ""), context, kind, reason))

    def _drop_locked(self, conn: socket.socket) -> None:
        self._acked.pop(conn, None)
        self._buffers.pop(conn, None)
        self._identity.pop(conn, None)
        self._left.discard(conn)
        self._tickets.pop(conn, None)

    def _confirmed_dead(self, conn: socket.socket) -> bool:
        """True only when ``conn``'s ticket is unheld.

        A subscriber locks its ticket for as long as it runs, released only by the
        kernel tearing down its file table on exit -- any exit, including a signal
        that skips every other cleanup path, so successfully acquiring it here
        proves that it's dead.
        We cannot confirm in other cases, so the answer is conservative on them.
        """
        path = self._tickets.get(conn)
        if path is None:
            return False
        try:
            with open(path, "rb") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return False
                fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            return False
        return True

    @property
    def taints(self) -> list[Taint]:
        """``(process_uid, test, kind, reason)`` for coverage the channel doubts.

        Two different sources feed the taints:

        - ``_taints`` entries record a direct, scoped outcome: the record for one
         named test cannot be trusted.

        - ``_lost`` entries, expanded here into ``TAINT_STOPPED_RECORDING``, cover a
         fan-out: failing to ACK, never receiving a context, or an EOF that isn't a
         confirmed kill all mean coverage for every test from that point on is not
         properly attributed, and none of them can be named individually.
        """
        with self._lock:
            taints = set(self._taints)
            for process_uid, superseded, sequence in self._lost:
                if superseded is not None:
                    taints.add((process_uid, superseded, TAINT_ATTRIBUTION, TAINT_UNACKNOWLEDGED))
                taints.update(
                    (process_uid, context, TAINT_INCOMPLETE, TAINT_STOPPED_RECORDING)
                    for announced_sequence, context in self._announced
                    if announced_sequence >= sequence
                )
            return sorted(taints)

    def announce(self, nodeid: str, ack_timeout: float = _DEFAULT_ACK_TIMEOUT) -> bool:
        """Publish a new context and, by default, wait for every consumer to confirm it."""
        with self._lock:
            if self._stopped:
                return True
            self._sequence += 1
            sequence = self._sequence
            # A subscriber that misses this announcement keeps recording under the
            # context it still believes is current, so that is the one whose rows
            # collect the next test's work.
            superseded = self._context
            self._context = nodeid
            self._announced.append((sequence, nodeid))
            for conn in list(self._acked):
                if not _send(conn, f"{_CONTEXT} {sequence} {nodeid}"):
                    self._drop_locked(conn)
        if not ack_timeout:
            return True
        return self.wait_acks(sequence, ack_timeout, taint_context=superseded)

    def wait_acks(
        self,
        sequence: int,
        timeout: float = _DEFAULT_ACK_TIMEOUT,
        taint_context: Optional[str] = None,
    ) -> bool:
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
                        # Expanded when the taints are read: what this subscriber goes
                        # on to miss is whatever gets announced after this point.
                        self._lost.append((self._identity.get(conn, ""), taint_context, sequence))
                        self._drop_locked(conn)
                        try:
                            # Closing prompts the subscriber's final save, which a
                            # signal would not; it stops recording instead of
                            # attributing later work to a test it has fallen behind.
                            conn.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return False
                self._progress.wait(remaining)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._acked)

    def close(self, drain_timeout: float = _DEFAULT_DRAIN_TIMEOUT) -> bool:
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
            for conn in list(self._acked):
                self._taint_locked(conn, TAINT_INCOMPLETE, TAINT_NOT_DRAINED)
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
        except OSError:
            pass
        shutil.rmtree(self._ticket_dir, ignore_errors=True)
        return drained


class ContextSubscriber:
    """Consumer: applies the current context, then every announcement until STOP."""

    def __init__(
        self,
        conn: socket.socket,
        on_context: Callable[[str], None],
        on_stop: Callable[[], None],
        on_flush: Optional[Callable[[], None]] = None,
    ) -> None:
        self._conn = conn
        self._on_context = on_context
        self._on_stop = on_stop
        self._on_flush = on_flush
        self._buffer = b""
        self._thread: Optional[threading.Thread] = None
        self._ticket_fh: Optional[object] = None

    @classmethod
    def subscribe(
        cls,
        address: Optional[str] = None,
        *,
        on_context: Callable[[str], None],
        on_stop: Callable[[], None],
        on_flush: Optional[Callable[[], None]] = None,
        identity: str = "",
        timeout: float = 2.0,
    ) -> Optional["ContextSubscriber"]:
        """Join the channel and apply the current context, or return None.

        The first frame is read here rather than on the reader thread, so a caller that
        returns from ``subscribe`` is already on the right context. ``identity`` names
        the leaf database this process writes, letting the producer attribute a taint to
        the coverage it belongs to. ``on_flush``, if given, runs after ``on_context`` and
        before the ACK for it goes out, so the producer never sees a test's context
        superseded until this subscriber's coverage for it is durably on disk -- see
        ``_pump``.
        """
        address = address or announced_address()
        if not address:
            return None
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(timeout)
        try:
            conn.connect(address)
        except OSError as exc:
            conn.close()
            # Reached only when an address was announced, so nothing listening is a
            # failure rather than the "nobody told me" case handled above.
            raise ChannelError(f"could not join the context channel at {address}") from exc
        if identity:
            _send(conn, f"{_IDENTITY} {identity}")
        subscriber = cls(conn, on_context, on_stop, on_flush)
        if not subscriber._pump(block=True):
            conn.close()
            return None
        conn.settimeout(None)
        subscriber._thread = threading.Thread(
            target=subscriber._run, daemon=True, name="cbts-context-subscriber"
        )
        subscriber._thread.start()
        return subscriber

    def _pump(self, block: bool = False) -> bool:
        """Read one chunk and dispatch its frames; False once the channel is finished."""
        try:
            chunk = self._conn.recv(4096)
        except OSError as exc:
            _report_unexpected("lost the context channel", exc)
            return False
        if not chunk:
            # The producer exited without a STOP; treat it as one so coverage is saved.
            self._on_stop()
            return False
        self._buffer, frames = _frames(self._buffer, chunk)
        for frame in frames:
            kind, _, payload = frame.partition(" ")
            if kind == _TICKET:
                self._hold_ticket(payload)
                continue
            sequence, _, nodeid = payload.partition(" ")
            if kind == _CONTEXT:
                self._on_context(nodeid)
                # Flush before ACK-ing: once the producer sees this ACK it may conclude
                # the superseded test is durably recorded (see ``TAINT_SUBSCRIBER_DEAD``,
                # which trusts everything before the current context), so that has to be
                # true by the time the ACK is sent, not just eventually via periodic save.
                if self._on_flush is not None:
                    self._on_flush()
            elif kind == _STOP:
                self._on_stop()
                _send(self._conn, f"{_ACK} {sequence}")
                return False
            else:
                continue
            _send(self._conn, f"{_ACK} {sequence}")
        del block
        return True

    def _hold_ticket(self, path: str) -> None:
        """Create and lock our ticket, held as long as this process is alive.

        This allows the server to reliably tell whether we are still alive.
        We let kernel's cleanup to set the signal by releasing the file lock,
        so we never need to release it ourselves.
        """
        try:
            fh = open(path, "wb")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            _report_unexpected("could not hold the context channel ticket", exc)
            return
        self._ticket_fh = fh

    def _run(self) -> None:
        while self._pump():
            pass
        self.close()

    def close(self) -> None:
        """Leave the channel normally, waking our own reader and signalling EOF to the producer."""
        _send(self._conn, _LEAVE)
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
