# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Liveness connections from Wide-EP ranks and node supervisors to the membership authority."""

import functools
import os
import select
import socket
import threading
import time
import traceback
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import IO, Optional

from tensorrt_llm.logger import logger

from .records import (
    AUTHORITY_SILENCE_TIMEOUT_S,
    DEAD_TIMEOUT_S,
    HELLO_TIMEOUT_S,
    REPORT_INTERVAL_S,
    MembershipRecord,
    RankReport,
    message_bytes,
)
from .rendezvous import LIVENESS_KEY, rendezvous_address, rendezvous_host_port
from .signed_store import SignedStore, shared_secret_digest

_MAX_LINE_BYTES = 1 << 20
_KEEPALIVE_PROBES = 6
_AUTHORITY_WAIT_WARNING_S = 30.0


class Lines:
    """Line framing for one side of a liveness connection."""

    def __init__(self) -> None:
        self._buffer = b""

    @staticmethod
    def encode(payload: bytes) -> bytes:
        return payload + b"\n"

    def decode(self, data: bytes) -> list[bytes]:
        """Return the payloads of the lines that ``data`` completes."""
        *lines, self._buffer = (self._buffer + data).split(b"\n")
        if len(self._buffer) > _MAX_LINE_BYTES:
            raise ValueError("a liveness line is too long")
        return lines


@dataclass
class LivenessConnection:
    """A process's liveness connection to the authority, closed at the end of a ``with`` block."""

    sock: socket.socket
    lines: Lines

    def __enter__(self) -> "LivenessConnection":
        return self

    def __exit__(self, *exc_info) -> None:
        self.sock.close()


def configure_connection(sock: socket.socket) -> None:
    """Set up a liveness connection's socket.

    It sends each record at once and fails once its peer is silent for ``DEAD_TIMEOUT_S``.
    """
    # Both ends send small records, which Nagle's algorithm would hold until the peer's delayed
    # acknowledgment.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    idle_s = int(DEAD_TIMEOUT_S / 2)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle_s)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, idle_s // _KEEPALIVE_PROBES)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, _KEEPALIVE_PROBES)
    # Keepalive probes start only on an idle connection, so this bounds a send the peer never
    # acknowledges.
    if hasattr(socket, "TCP_USER_TIMEOUT"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, int(DEAD_TIMEOUT_S * 1000))


@functools.lru_cache(maxsize=1)
def _closed_in_forked_child() -> "weakref.WeakSet[socket.socket | IO[bytes]]":
    resources = weakref.WeakSet()

    def close_all() -> None:
        for resource in list(resources):
            resource.close()

    os.register_at_fork(after_in_child=close_all)
    return resources


def close_in_forked_child(resource: socket.socket | IO[bytes]) -> None:
    """Close ``resource`` in every child that this process forks, without keeping it alive."""
    _closed_in_forked_child().add(resource)


def _warn_while_no_authority(store: SignedStore) -> None:
    """Log once if the authority has not published its liveness address within the warning time."""
    from torch.distributed import DistStoreError

    try:
        store.wait([LIVENESS_KEY], timedelta(seconds=_AUTHORITY_WAIT_WARNING_S))
    except DistStoreError:
        logger.warning(
            f"No membership authority has published its liveness address in the rendezvous "
            f"store at {store.host}:{store.port} after "
            f"{_AUTHORITY_WAIT_WARNING_S:.0f} s. Still waiting"
        )


def hello_token() -> str:
    """Return the token by which the authority admits a process of this deployment."""
    return shared_secret_digest(b"liveness hello")


def _open_connection(store: SignedStore, announcement: str, timeout_s: float) -> LivenessConnection:
    """Open a connection to the authority and send the hello that carries ``announcement``."""
    _warn_while_no_authority(store)
    address = rendezvous_host_port(store.get(LIVENESS_KEY).decode())
    sock = socket.create_connection(address, timeout=timeout_s)
    configure_connection(sock)
    lines = Lines()
    sock.sendall(lines.encode(f"{hello_token()} {announcement}".encode()))
    sock.settimeout(None)
    return LivenessConnection(sock, lines)


def open_liveness_connection(
    store: SignedStore, rank: int, replacement: bool = False
) -> LivenessConnection:
    """Open this process's liveness connection to the authority for ``rank``.

    Args:
        store: The rendezvous store, where the authority publishes its liveness address.
        rank: The rank this process runs.
        replacement: Whether this process replaces a failed process of ``rank``.
    """
    connection = _open_connection(store, f"{rank} {int(replacement)}", HELLO_TIMEOUT_S)
    close_in_forked_child(connection.sock)
    return connection


def open_node_supervisor_connection(
    store: SignedStore,
    node: int,
    ranks: list[int],
    host: str,
    timeout_s: float = HELLO_TIMEOUT_S,
) -> LivenessConnection:
    """Open a node supervisor's connection for ``node``. Closing it fails ``ranks``.

    Args:
        store: The rendezvous store, where the authority publishes its liveness address.
        node: The node that the supervisor runs on.
        ranks: The ranks that the supervisor starts on its node.
        host: The node's host name, which the authority logs.
        timeout_s: How long to wait for the connection to the authority.
    """
    return _open_connection(store, f"node {node} {','.join(map(str, ranks))} {host}", timeout_s)


def open_store(address: str, timeout_s: float, secret: Optional[str] = None) -> SignedStore:
    """Open a client for the store that the authority hosts at ``host:port``.

    The client signs and verifies values with ``secret`` (hex), by default ``SHARED_SECRET_ENV``.
    """
    from torch.distributed import TCPStore

    host, port = rendezvous_host_port(address)
    store = TCPStore(
        host,
        port,
        is_master=False,
        timeout=timedelta(seconds=timeout_s),
        wait_for_workers=False,
    )
    return SignedStore(store, secret)


def rendezvous_store(timeout_s: float = DEAD_TIMEOUT_S) -> SignedStore:
    """Open a rendezvous store client from this process's environment.

    Raises:
        ValueError: The environment holds no rendezvous address or shared secret.
    """
    return open_store(rendezvous_address(), timeout_s)


class MembershipClient:
    """A rank's thread that keeps its liveness connection, delivers records and sends reports.

    The thread ends once a record drops this process from the membership, and leaves the connection
    open: the authority admits a replacement for the rank only once it closes, when the owner calls
    :meth:`stop` or exits.
    """

    def __init__(
        self,
        rank: int,
        store: SignedStore,
        on_record: Callable[[MembershipRecord], None],
        replacement: bool = False,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Prepare the client. :meth:`start` connects to the authority.

        Args:
            rank: The rank this process runs.
            store: The rendezvous store, where the authority publishes its liveness address.
            on_record: Called on the client thread with each record the authority sends.
            replacement: Whether this process replaces a failed process of ``rank``.
            on_error: Called on the client thread with the error that ended it, if any.
        """
        self.rank = rank
        self._store = store
        self._on_record = on_record
        self._on_tick: Optional[Callable[[], None]] = None
        self._on_error = on_error
        self._replacement = replacement
        self._report = RankReport()
        self.record: Optional[MembershipRecord] = None
        self.incarnation: Optional[int] = None
        self._lock = threading.Lock()
        self._delivery = threading.Lock()
        self._stop = threading.Event()
        self._wake_read, self._wake_write = os.pipe()
        os.set_blocking(self._wake_read, False)
        os.set_blocking(self._wake_write, False)
        self._thread = threading.Thread(
            target=self._run, name=f"MembershipClient{rank}", daemon=True
        )
        self._sock: Optional[socket.socket] = None
        self._lines: Optional[Lines] = None
        self._received: list[bytes] = []

    def start(self) -> None:
        """Connect, take the authority's first record and start the thread.

        The first record sets ``record`` and ``incarnation`` at once. The thread also delivers it
        to ``on_record``.
        """
        connection = open_liveness_connection(self._store, self.rank, self._replacement)
        self._sock, self._lines = connection.sock, connection.lines
        try:
            self._sock.settimeout(HELLO_TIMEOUT_S)
            while not self._received:
                data = self._sock.recv(1 << 16)
                if not data:
                    raise ConnectionError(
                        f"the membership authority refused this process of rank {self.rank}"
                    )
                self._received = self._lines.decode(data)
            self._sock.settimeout(None)
            self.record = MembershipRecord.model_validate_json(self._received[0])
            self.incarnation = self.record.incarnations[self.rank]
            self._send_report()
        except BaseException:
            # The authority admits a replacement for this rank only once the connection closes.
            self._sock.close()
            self._sock = None
            raise
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake()
        if self._thread.ident is not None:
            self._thread.join()
        if self._sock is not None:
            self._sock.close()
        # A report() racing this stop must not write to a closed or reused descriptor.
        with self._lock:
            os.close(self._wake_read)
            os.close(self._wake_write)
            self._wake_write = None

    def bind(
        self,
        on_record: Callable[[MembershipRecord], None],
        on_tick: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Replace the callbacks, delivering the newest record to the new ``on_record`` at once."""
        with self._delivery:
            self._on_record, self._on_tick, self._on_error = on_record, on_tick, on_error
            if self.record is not None:
                on_record(self.record)

    def report(self, **changes) -> None:
        """Update report fields and wake the thread to send it.

        Raises:
            pydantic.ValidationError: ``changes`` names a field the report lacks or has a value of
                the wrong type.
        """
        with self._lock:
            self._report = RankReport.model_validate({**self._report.model_dump(), **changes})
        self._wake()

    def _wake(self) -> None:
        with self._lock:
            if self._wake_write is None:
                return
            try:
                os.write(self._wake_write, b"\0")
            except BlockingIOError:
                pass

    def _send_report(self) -> None:
        with self._lock:
            report = self._report
        self._sock.sendall(self._lines.encode(message_bytes(report)))

    def _run(self) -> None:
        try:
            self._serve()
            return
        except (OSError, ValueError) as error:
            logger.error(f"Membership client of rank {self.rank} lost the authority: {error!r}")
            failure = error
        except _CallbackError as error:
            failure = error.__cause__
            logger.error(
                f"The {error} callback of the membership client of rank {self.rank} failed:\n"
                + "".join(traceback.format_exception(failure))
            )
        # The thread ends here, so any other error must reach the owner too, or the rank would
        # keep a connection that no longer reports.
        except Exception as error:
            failure = error
            logger.error(
                f"Membership client of rank {self.rank} failed:\n"
                + "".join(traceback.format_exception(failure))
            )
        self._sock.close()
        if self._on_error is not None:
            self._on_error(failure)

    def _serve(self) -> None:
        listed = False
        tick_s = heard_s = time.monotonic()
        lines, self._received = self._received, []
        while not self._stop.is_set():
            # A heartbeat line is empty.
            for line in filter(None, lines):
                record = MembershipRecord.model_validate_json(line)
                with self._delivery:
                    self.record = record
                    _call("record", self._on_record, record)
                member = record.is_member(self.rank, self.incarnation)
                if listed and not member:
                    return
                listed = member
            watched = [self._sock, self._wake_read]
            readable, _, _ = select.select(watched, [], [], max(0.0, tick_s - time.monotonic()))
            if (
                self._sock not in readable
                and time.monotonic() - heard_s > AUTHORITY_SILENCE_TIMEOUT_S
            ):
                # This thread can wait long for a CPU or the GIL after its select returns, while the
                # authority's lines arrive. Look again before taking the authority as silent.
                readable, _, _ = select.select(watched, [], [], 0.0)
                if self._sock not in readable:
                    raise TimeoutError(
                        "the membership authority sent nothing for "
                        f"{AUTHORITY_SILENCE_TIMEOUT_S:g} s"
                    )
            lines = []
            if self._sock in readable:
                data = self._sock.recv(1 << 16)
                if not data:
                    raise ConnectionError("the membership authority closed the liveness connection")
                lines = self._lines.decode(data)
                if lines:
                    heard_s = time.monotonic()
            due = time.monotonic() >= tick_s
            if due:
                tick_s = time.monotonic() + REPORT_INTERVAL_S
                if self._on_tick is not None:
                    _call("tick", self._on_tick)
            if due or self._wake_read in readable:
                try:
                    os.read(self._wake_read, 1 << 12)
                except BlockingIOError:
                    pass
                self._send_report()


class _CallbackError(Exception):
    """Raised when a callback of :class:`MembershipClient` fails, naming the callback."""


def _call(name: str, callback: Callable[..., None], *args) -> None:
    try:
        callback(*args)
    except Exception as error:
        raise _CallbackError(name) from error
