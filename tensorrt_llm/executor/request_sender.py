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

"""Single-owner command delivery and command-connection loss detection."""

import collections
import itertools

# Only serializes executor commands; this owner does not deserialize input.
import pickle  # nosec B403
import sys
import threading
import time
import weakref
from queue import Queue

import zmq
from zmq.utils.monitor import recv_monitor_message

from .ipc import IpcQueue
from .request import _COMMAND_QUEUE_HWM, CancellingRequest
from .utils import EngineDeadError, EngineUnavailableError

_HANDSHAKE_FAILED_EVENTS = (
    zmq.EVENT_HANDSHAKE_FAILED_NO_DETAIL
    | zmq.EVENT_HANDSHAKE_FAILED_PROTOCOL
    | zmq.EVENT_HANDSHAKE_FAILED_AUTH
)
_MONITOR_EVENTS = (
    zmq.EVENT_ACCEPTED
    | zmq.EVENT_CONNECTED
    | zmq.EVENT_HANDSHAKE_SUCCEEDED
    | _HANDSHAKE_FAILED_EVENTS
    | zmq.EVENT_DISCONNECTED
)


class _CommandConnection:
    """Tell a command-peer disconnect apart from an unrelated connection.

    A bound command socket also accepts connections that never carry commands,
    such as a port probe or a second ZMTP peer that PAIR rejects. Commands are
    only queued to a peer after its handshake, and the first completed handshake
    is the command peer: PAIR attaches no later peer and a connecting socket has
    one connection. Monitor events carry the file descriptor for accept, connect
    and disconnect, but not for a completed handshake. A completion during
    concurrent handshakes therefore marks every candidate as the possible peer.
    IpcQueue uses one libzmq I/O thread, so a handshake failure and the
    disconnect it causes are adjacent events; that pair clears one candidate.
    Some rejected connections, such as a TLS probe or a ZMTP peer of another
    socket type, disconnect without a handshake failure. For such a candidate,
    the caller reports whether the socket still has an attached, writable peer.
    """

    def __init__(self) -> None:
        self._handshaking: set[int] = set()
        # Empty until a handshake completes; None if no candidate was observed.
        self._peer: set[int] | None = set()
        self._handshake_failed = False

    def observe(self, event: int, value: int, peer_attached=lambda: False) -> bool:
        """Record one monitor event; return whether the command peer was lost.

        Args:
            event: Socket monitor event.
            value: Event value; the file descriptor for connection events.
            peer_attached: Whether the socket still has a writable peer. It is
                consulted only for an ambiguous candidate's disconnect.
        """
        handshake_failed, self._handshake_failed = (
            self._handshake_failed,
            bool(event & _HANDSHAKE_FAILED_EVENTS),
        )
        if event in (zmq.EVENT_ACCEPTED, zmq.EVENT_CONNECTED):
            self._handshaking.add(value)
        elif event == zmq.EVENT_HANDSHAKE_SUCCEEDED:
            if self._peer is not None and not self._peer:
                self._peer = set(self._handshaking) or None
            if self._peer is not None and len(self._handshaking) == 1:
                self._handshaking.clear()
        elif event == zmq.EVENT_DISCONNECTED:
            self._handshaking.discard(value)
            if self._peer is None:
                return True
            if value in self._peer:
                if len(self._peer) > 1 and (handshake_failed or peer_attached()):
                    self._peer.discard(value)
                    return False
                return True
        return False


class _RequestSender:
    """Own command sends and closure, including the worker shutdown message.

    Connected backpressure retains queued commands without declaring the engine
    dead. Losing an established direct connection makes delivery uncertain, so
    pending requests fail rather than replaying commands after reconnection.

    With ``direct_send``, callers can write queued commands themselves. CPython
    can deny a waiting thread the GIL for as long as another thread keeps
    briefly releasing it, as a busy asyncio loop does, so admission and loss
    detection must not depend on scheduling the owner thread. Every socket and
    monitor use then happens under the condition lock, without waiting while
    holding it, which provides the memory barrier libzmq requires when a socket
    changes threads. The owner retries non-blocking writes during backpressure
    and connection setup, enforces their deadlines, and performs shutdown.

    Unbatched callers write each command at once. With batching, the owner
    coalesces queued commands into one message; a caller that finds admission
    full writes the backlog as one message instead of rejecting.

    Code under the condition lock avoids calls that release the GIL where an
    equivalent exists: it checks socket readiness with ``ZMQ_EVENTS`` rather
    than ``zmq_poll`` and wakes the owner only when the owner has work. A thread
    that releases the GIL while holding the lock must wait for the GIL again
    before the lock is free; with another CPU-bound thread, each such release
    can hold every caller for several switch intervals. For the same reason
    the direct owner waits on an event, not the condition, and stays off the
    lock while callers keep checking the connection.
    """

    MAX_BATCH_SIZE = 1024
    MAX_PENDING_REQUESTS = 1024
    MAX_PENDING_CONTROLS = 1024
    MAX_INFLIGHT_BATCHES = _COMMAND_QUEUE_HWM
    SEND_TIMEOUT = 0.05
    # Callers check the connection on every send; the owner covers idle time.
    IDLE_CHECK_INTERVAL = 0.2
    CONNECT_TIMEOUT = 5.0

    def __init__(
        self,
        request_queue: IpcQueue,
        error_queue: Queue,
        *,
        batching: bool = True,
        on_failure: weakref.WeakMethod | None = None,
        connect_timeout: float = CONNECT_TIMEOUT,
        stop_workers_on_failure: bool = False,
        direct_send: bool = False,
    ) -> None:
        self._request_queue = request_queue
        self._error_queue = error_queue
        self._batching = batching
        self._on_failure = on_failure
        self._connect_timeout = connect_timeout
        self._pending = collections.deque()
        self._pending_requests = 0
        self._pending_controls = 0
        self._capacity_waiters = 0
        # The direct owner waits on this event rather than the condition.
        self._owner_wake = threading.Event()
        self._connection_checked = float("-inf")
        self._pending_cancel_ids: set[int] = set()
        self._stop_workers_on_failure = stop_workers_on_failure
        self._cond = threading.Condition()
        self._closed = False
        self._send_sentinel = False
        self._shutdown_cancel_ids: tuple[int, ...] = ()
        self._terminal_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._is_finalizing = sys.is_finalizing
        self._monitor: zmq.Socket | None = None
        self._connection = _CommandConnection()
        self._connection_lost: ConnectionError | None = None
        self._connected = False
        self._direct_send = direct_send
        self._close_deadline = float("inf")
        self._ready = threading.Event()
        with self._cond:
            self._start()
        self._ready.wait()
        if self._terminal_error is not None:
            raise EngineDeadError(self._terminal_error)

    @property
    def available(self) -> bool:
        """Whether the sender is operational, independent of queue congestion."""
        with self._cond:
            return (
                self._terminal_error is None and self._connection_lost is None and not self._closed
            )

    def _start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="proxy_request_sender"
            )
            self._thread.start()

    def send(
        self,
        payload: bytes,
        *,
        control: bool = False,
        cancellation_id: int | None = None,
        wait_for_capacity: bool = False,
    ) -> None:
        """Queue a captured command, rejecting work that cannot be admitted."""
        with self._cond:
            self._check_admission()
            if cancellation_id is not None and cancellation_id in self._pending_cancel_ids:
                return
            limit = self.MAX_PENDING_CONTROLS if control else self.MAX_PENDING_REQUESTS
            while (
                cancellation_id is None
                and (self._pending_controls if control else self._pending_requests) >= limit
            ):
                self._write_pending_locked()
                if (self._pending_controls if control else self._pending_requests) < limit:
                    break
                if not wait_for_capacity:
                    raise EngineUnavailableError("Executor command queue is full")
                self._capacity_waiters += 1
                try:
                    self._cond.wait(self.SEND_TIMEOUT if self._direct_send else None)
                finally:
                    self._capacity_waiters -= 1
                self._check_admission()
            self._pending.append((payload, control, cancellation_id))
            if cancellation_id is not None:
                self._pending_cancel_ids.add(cancellation_id)
            elif control:
                self._pending_controls += 1
            else:
                self._pending_requests += 1
            if not self._batching:
                self._write_pending_locked()
            # A command left queued needs the owner, and a write may have made
            # room for a caller waiting for capacity. Waking the owner after it
            # has nothing to do costs every caller GIL handoffs under load.
            if self._pending or self._capacity_waiters:
                self._wake_locked()

    def _write_pending_locked(self, *, owner: bool = False) -> None:
        """Write queued commands in order without waiting for the socket."""
        if not self._direct_send or self._terminal_error is not None:
            return
        if self._closed and not owner:
            return
        self._poll_connection_locked()
        if self._connection_lost is not None:
            # The owner fails accepted work; nothing may follow a lost peer.
            self._wake_locked()
            return
        size = self.MAX_BATCH_SIZE if self._batching else 1
        while self._pending:
            items = list(itertools.islice(self._pending, size))
            try:
                # A zero timeout checks writability without zmq_poll, which
                # would release the GIL while the condition lock is held.
                self._request_queue.put([item[0] for item in items], timeout=0)
            except zmq.Again:
                # The owner retries during backpressure or connection setup.
                return
            except Exception:
                if owner:
                    raise
                # The owner retries the same commands and records the failure.
                self._wake_locked()
                return
            for _ in items:
                self._pending.popleft()
            self._connected = True
            self._mark_taken(items)
            self._mark_sent(items)

    def _wake_locked(self) -> None:
        """Wake callers waiting on the condition and the direct owner."""
        self._cond.notify_all()
        self._owner_wake.set()

    def _mark_taken(self, items) -> None:
        for _, control, cancellation_id in items:
            if cancellation_id is not None:
                continue
            elif control:
                self._pending_controls -= 1
            else:
                self._pending_requests -= 1

    def _mark_sent(self, items) -> None:
        for _, _, cancellation_id in items:
            if cancellation_id is not None:
                self._pending_cancel_ids.discard(cancellation_id)

    def _check_admission(self) -> None:
        if self._terminal_error is not None:
            raise EngineDeadError(self._terminal_error)
        if self._connection_lost is not None:
            raise EngineDeadError(self._connection_lost)
        if self._closed:
            raise EngineUnavailableError("Executor command sender is stopping")

    def fail(self, error: BaseException) -> None:
        """Stop delivery after an independently confirmed terminal failure."""
        with self._cond:
            if self._terminal_error is None:
                self._terminal_error = error
            self._send_sentinel |= self._stop_workers_on_failure
            self._pending.clear()
            self._pending_cancel_ids.clear()
            self._pending_controls = self._pending_requests = 0
            if not self._is_finalizing():
                self._start()
            self._wake_locked()

    def close(
        self,
        timeout: float | None = None,
        *,
        send_sentinel: bool = False,
        cancel_ids: tuple[int, ...] = (),
    ) -> None:
        """Request bounded owner-thread shutdown; the owner writes final commands."""
        if self._is_finalizing():
            return
        with self._cond:
            if not self._closed:
                self._shutdown_cancel_ids = cancel_ids
            self._closed = True
            if timeout is not None:
                self._close_deadline = min(self._close_deadline, time.monotonic() + timeout)
            self._send_sentinel |= send_sentinel
            self._start()
            thread = self._thread
            self._wake_locked()
        if thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                self._record_failure(TimeoutError("Executor command sender shutdown timed out"))

    def _peer_attached(self) -> bool:
        return bool(self._request_queue.socket.getsockopt(zmq.EVENTS) & zmq.POLLOUT)

    def _poll_connection_locked(self) -> None:
        self._connection_checked = time.monotonic()
        # ZMQ_EVENTS processes pending socket commands like zmq_poll, but
        # without releasing the GIL while the condition lock is held.
        while self._monitor.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            message = recv_monitor_message(self._monitor)
            event = message["event"]
            if event == zmq.EVENT_HANDSHAKE_SUCCEEDED:
                self._connected = True
            lost = self._connection.observe(event, message["value"], self._peer_attached)
            if lost and not self._closed and self._connection_lost is None:
                self._connection_lost = ConnectionError(
                    "Executor command connection was lost; request delivery is uncertain"
                )

    def _check_terminal(self) -> None:
        with self._cond:
            self._poll_connection_locked()
            if self._terminal_error is not None:
                raise EngineDeadError(self._terminal_error)
            if self._connection_lost is not None:
                raise self._connection_lost

    def _record_failure(self, error: BaseException) -> None:
        with self._cond:
            report = self._terminal_error is None
        self.fail(error)
        callback = self._on_failure() if self._on_failure is not None else None
        if report and callback is not None:
            callback(error)
        elif report:
            self._error_queue.put_nowait(error)

    def _put_until(self, message: object, deadline: float = float("inf")) -> None:
        connect_deadline = time.monotonic() + self._connect_timeout
        while True:
            self._check_terminal()
            remaining = (
                min(
                    deadline,
                    self._close_deadline,
                    float("inf") if self._connected else connect_deadline,
                )
                - time.monotonic()
            )
            if remaining <= 0:
                raise TimeoutError("Executor command connection or shutdown timed out")
            try:
                self._request_queue.put(message, timeout=min(self.SEND_TIMEOUT, remaining))
                self._connected = True
                self._check_terminal()
                return
            except zmq.Again:
                with self._cond:
                    self._cond.wait(timeout=min(self.SEND_TIMEOUT, remaining))

    def _take_batch(self) -> list[tuple[bytes, bool, int | None]] | None:
        with self._cond:
            while not self._pending and not self._closed and self._terminal_error is None:
                self._cond.wait(self.SEND_TIMEOUT)
                self._check_terminal()
            if self._terminal_error is not None:
                raise EngineDeadError(self._terminal_error)
            if not self._pending:
                return None
            size = self.MAX_BATCH_SIZE if self._batching else 1
            items = [self._pending.popleft() for _ in range(min(size, len(self._pending)))]
            self._mark_taken(items)
            self._cond.notify_all()
            return items

    def _run_direct(self) -> None:
        """Retry queued writes, leaving the socket to callers between attempts.

        The owner waits on its own event because a condition wait always takes
        the lock back, competing with callers. With nothing queued, it checks
        the connection only after callers have not checked it for
        ``IDLE_CHECK_INTERVAL``, so it does not compete with active callers for
        the lock or the GIL and still detects the loss of an idle connection.
        """
        connect_deadline = None
        while True:
            # Cleared before the state is read, so no wake-up is lost.
            self._owner_wake.clear()
            with self._cond:
                self._check_terminal()
                self._write_pending_locked(owner=True)
                retry = bool(self._pending)
                if not retry:
                    if self._closed:
                        return
                    connect_deadline = None
                else:
                    now = time.monotonic()
                    if not self._connected:
                        if connect_deadline is None:
                            connect_deadline = now + self._connect_timeout
                        if now >= connect_deadline:
                            raise TimeoutError("Executor command connection timed out")
                    if now >= self._close_deadline:
                        raise TimeoutError("Executor command sender shutdown timed out")
            # A caller that leaves a command queued, closes or fails the sender
            # sets the event. Queued commands are not read here: a caller can
            # hold one while it writes it, and waking for it would only wait
            # for that caller's lock.
            while not self._owner_wake.wait(
                self.SEND_TIMEOUT if retry else self.IDLE_CHECK_INTERVAL
            ):
                if retry or time.monotonic() - self._connection_checked >= self.IDLE_CHECK_INTERVAL:
                    break

    def _run(self) -> None:
        try:
            # Connect-side sockets must not report successful delivery by
            # buffering commands for a peer that has not connected yet.
            self._request_queue.socket.setsockopt(zmq.IMMEDIATE, 1)
            self._monitor = self._request_queue.socket.get_monitor_socket(events=_MONITOR_EVENTS)
            self._request_queue.setup_lazily()
            self._ready.set()
            if self._direct_send:
                self._run_direct()
            while (batch := self._take_batch()) is not None:
                self._put_until([payload for payload, _, _ in batch])
                with self._cond:
                    self._mark_sent(batch)
                    self._cond.notify_all()
        except Exception as error:
            self._record_failure(error)
        finally:
            self._ready.set()
            with self._cond:
                send_sentinel = self._send_sentinel
            if self._terminal_error is None:
                try:
                    # Attached frontends cancel their own live requests even
                    # when ordinary control capacity was already exhausted.
                    for start in range(0, len(self._shutdown_cancel_ids), self.MAX_BATCH_SIZE):
                        ids = self._shutdown_cancel_ids[start : start + self.MAX_BATCH_SIZE]
                        self._put_until([pickle.dumps(CancellingRequest(i)) for i in ids])
                except (zmq.ZMQError, TimeoutError, EngineDeadError) as error:
                    self._record_failure(error)
            if send_sentinel:
                try:
                    if self._terminal_error is None:
                        self._put_until(None, time.monotonic() + 1.0)
                    else:
                        self._request_queue.put(None, timeout=self.SEND_TIMEOUT)
                except (zmq.ZMQError, TimeoutError, EngineDeadError) as error:
                    self._record_failure(error)
            try:
                if self._monitor is not None:
                    try:
                        self._request_queue.socket.disable_monitor()
                    finally:
                        self._monitor.close(linger=0)
            finally:
                linger = 0 if self._terminal_error is not None else 1000
                self._request_queue.socket.setsockopt(zmq.LINGER, linger)
                self._request_queue.close()
