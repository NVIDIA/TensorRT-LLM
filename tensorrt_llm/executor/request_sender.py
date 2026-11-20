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
from collections.abc import Callable
from queue import Queue

import zmq
from zmq.utils.monitor import recv_monitor_message

from tensorrt_llm.logger import logger

from .ipc import CLOSE_LINGER_MS, IpcQueue
from .request import CancellingRequest
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
# Rank 0 stops reading commands at a None message.
_SENTINEL = pickle.dumps(None)
# A queued command: (pickled payload, counted against MAX_PENDING_REQUESTS,
# cancelled request id or None).
_Command = tuple[bytes, bool, int | None]


class _PeerGoneWhileClosing(Exception):
    """The command peer disconnected while the sender was writing final commands."""


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

    def observe(
        self, event: int, value: int, peer_attached: Callable[[], bool] = lambda: False
    ) -> bool:
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
    full writes the backlog as one message instead of rejecting. A message is
    serialized and signed with the condition lock released, so a large batch
    does not hold up callers that are queuing commands.

    Code under the condition lock avoids calls that release the GIL where an
    equivalent exists: it checks socket readiness with ``ZMQ_EVENTS`` rather
    than ``zmq_poll`` and wakes the owner only when the queue turns non-empty.
    A thread that releases the GIL while holding the lock must wait for the
    GIL again before the lock is free; with another CPU-bound thread, each such
    release can hold every caller for several switch intervals. For the same
    reason the direct owner waits on an event, not the condition, and stays
    off the lock while callers keep checking the connection.

    Admission bounds generation requests queued here, beyond the
    ``send_hwm()`` messages the socket buffers. Rank 0 reads its command
    socket without a receive limit, and libzmq moves messages to it without
    the GIL, so admission engages only while rank 0 is not connected, has
    stopped reading (for example, its process is frozen), or the connection
    cannot keep up. Commands that rank 0 has received but not yet processed
    are not bounded here.
    """

    MAX_BATCH_SIZE = 1024
    MAX_PENDING_REQUESTS = 1024
    # One batched message carries up to MAX_BATCH_SIZE commands, so a few
    # buffered messages already hold many commands.
    BATCHED_SEND_HWM = 16
    # How long a blocked write or wait lasts before connection loss, failure
    # and deadlines are checked again.
    POLL_INTERVAL = 0.05
    # Callers check the connection on every send; the owner covers idle time.
    IDLE_CHECK_INTERVAL = 0.2
    CONNECT_TIMEOUT = 5.0
    # Bound on queuing the stop sentinel in an orderly close.
    SENTINEL_TIMEOUT = 1.0

    @classmethod
    def send_hwm(cls, batching: bool) -> int:
        """Return the command socket's send high-water mark, in messages.

        Unbatched, every command is one message, so the socket buffers as many
        commands as admission queues.
        """
        return cls.BATCHED_SEND_HWM if batching else cls.MAX_PENDING_REQUESTS

    def __init__(
        self,
        request_queue: IpcQueue,
        error_queue: Queue,
        *,
        batching: bool = False,
        on_failure: weakref.WeakMethod | None = None,
        connect_timeout: float = CONNECT_TIMEOUT,
        stop_workers_on_failure: bool = True,
        direct_send: bool = True,
    ) -> None:
        """Start the owner thread and wait until it monitors the socket.

        Args:
            request_queue: Command socket to rank 0, created with
                ``send_hwm(batching)``. The sender uses and closes it.
            error_queue: Receives the first terminal failure when
                ``on_failure`` is None or its target no longer exists.
            batching: Coalesce queued commands into messages of up to
                ``MAX_BATCH_SIZE`` commands.
            on_failure: Called with the first terminal failure seen before
                ``close()``, instead of putting it on ``error_queue``.
            connect_timeout: Seconds a queued command waits for rank 0 to
                connect before the sender fails.
            stop_workers_on_failure: Send the worker-stop sentinel after a
                terminal failure. Only the frontend that owns the workers sets it.
            direct_send: Let callers write commands themselves. False keeps
                every socket use on the owner thread, which the
                ``TLLM_LLMAPI_ZMQ_DEBUG`` thread check requires.

        Raises:
            EngineDeadError: If the owner thread could not set up the socket.
        """
        self._request_queue = request_queue
        self._error_queue = error_queue
        self._batching = batching
        self._on_failure = on_failure
        self._connect_timeout = connect_timeout
        self._pending: collections.deque[_Command] = collections.deque()
        self._pending_requests = 0
        self._capacity_waiters = 0
        # Set while a thread writes queued commands, including while it
        # serializes them with the lock released. No other thread writes then,
        # which keeps commands in order.
        self._writing = False
        # The serialized message for the first queued commands, kept when the
        # socket was not writable: (number of commands, message).
        self._prepared: tuple[int, bytes] | None = None
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
        # Bound here because close() can run from a finalizer during
        # interpreter teardown, when module globals such as sys may already
        # be None.
        self._is_finalizing = sys.is_finalizing
        self._monitor: zmq.Socket | None = None
        self._connection = _CommandConnection()
        # The command connection was lost or a write failed. Any thread can
        # record it; the owner reports it.
        self._delivery_error: BaseException | None = None
        # A disconnect after close() began is expected (rank 0 exits after the
        # stop sentinel), so it is not a failure; it only ends final writes.
        self._peer_gone_while_closing = False
        self._connected = False
        self._direct_send = direct_send
        self._close_deadline = float("inf")
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="proxy_request_sender")
        self._thread.start()
        self._ready.wait()
        if self._terminal_error is not None:
            raise EngineDeadError(self._terminal_error)

    @property
    def available(self) -> bool:
        """Whether the sender is operational, independent of queue congestion."""
        with self._cond:
            return (
                self._terminal_error is None and self._delivery_error is None and not self._closed
            )

    def send(
        self,
        payload: bytes,
        *,
        control: bool = False,
        cancellation_id: int | None = None,
        wait_for_capacity: bool = False,
    ) -> None:
        """Queue a pickled command for rank 0, writing it at once when possible.

        Generation requests count against ``MAX_PENDING_REQUESTS``. Control
        commands and cancellations do not, so a full queue never rejects them.

        Args:
            payload: The pickled command, captured as it is now.
            control: A command whose callers already send one at a time, such
                as profile start and stop. It is not counted.
            cancellation_id: The id of the request that ``payload`` cancels.
                It is not counted, and while a cancellation for the same id is
                still queued this one is dropped as a duplicate.
            wait_for_capacity: For a counted request, wait for room instead of
                raising when the queue is full.

        Raises:
            EngineDeadError: The sender failed or the command connection was
                lost.
            EngineUnavailableError: The sender is closing, or the queue is full
                and ``wait_for_capacity`` is False.
        """
        counted = not control and cancellation_id is None
        with self._cond:
            self._check_admission()
            if cancellation_id is not None and cancellation_id in self._pending_cancel_ids:
                return
            while counted and self._pending_requests >= self.MAX_PENDING_REQUESTS:
                self._write_pending_locked()
                self._check_admission()
                if self._pending_requests < self.MAX_PENDING_REQUESTS:
                    break
                # Another thread writing queued commands is about to make room.
                if not wait_for_capacity and not self._writing:
                    raise EngineUnavailableError("Executor command queue is full")
                self._capacity_waiters += 1
                try:
                    self._cond.wait(self.POLL_INTERVAL if self._direct_send else None)
                finally:
                    self._capacity_waiters -= 1
                self._check_admission()
            was_empty = not self._pending
            self._pending.append((payload, counted, cancellation_id))
            if cancellation_id is not None:
                self._pending_cancel_ids.add(cancellation_id)
            self._pending_requests += counted
            if not self._batching:
                self._write_pending_locked()
            # Once the owner has seen queued commands it retries on its own, so
            # it needs waking only when the queue turns non-empty. Waking it
            # more often costs every caller GIL handoffs under load.
            if was_empty and self._pending:
                self._wake_locked()

    def _write_pending_locked(self, *, owner: bool = False) -> None:
        """Write queued commands in order without waiting for the socket.

        The caller holds the condition lock once. The lock is released while a
        message is serialized and signed, so other callers can queue commands
        meanwhile; they do not write until this thread is done.

        Args:
            owner: Raise a transport error to the owner thread instead of
                recording it for the owner to report.
        """
        if not self._direct_send or self._writing:
            return
        self._writing = True
        try:
            while self._pending and self._may_write_locked():
                if self._prepared is None:
                    size = self.MAX_BATCH_SIZE if self._batching else 1
                    count = min(size, len(self._pending))
                    payloads = [item[0] for item in itertools.islice(self._pending, count)]
                    self._cond.release()
                    try:
                        message = self._encode(payloads)
                    finally:
                        self._cond.acquire()
                    # A failure or a lost peer may have ended delivery meanwhile.
                    if not self._may_write_locked():
                        return
                    self._prepared = (count, message)
                count, message = self._prepared
                try:
                    # A zero timeout checks writability without zmq_poll, which
                    # would release the GIL while the condition lock is held.
                    self._request_queue.put_signed(message, timeout=0)
                except zmq.Again:
                    # The owner retries during backpressure or connection setup.
                    return
                except zmq.ZMQError as error:
                    if owner:
                        raise
                    # Nothing may follow a failed write; the owner reports it.
                    self._delivery_error = error
                    self._wake_locked()
                    return
                self._prepared = None
                items = [self._pending.popleft() for _ in range(count)]
                self._connected = True
                self._mark_taken(items)
                self._mark_sent(items)
                # Waiters can queue now, while the next message is serialized.
                if self._capacity_waiters:
                    self._cond.notify_all()
        finally:
            self._writing = False
            if self._capacity_waiters:
                self._cond.notify_all()

    def _may_write_locked(self) -> bool:
        """Check the connection; return whether queued commands may be written."""
        if self._terminal_error is not None:
            return False
        self._poll_connection_locked()
        if self._delivery_error is not None or self._peer_gone_while_closing:
            # The owner ends delivery; nothing may follow a lost peer.
            self._wake_locked()
            return False
        return True

    def _encode(self, payloads: list[bytes]) -> bytes:
        """Return one signed message: a lone command as itself, several as a list."""
        pickled = payloads[0] if len(payloads) == 1 else pickle.dumps(payloads)
        return self._request_queue.sign_pickled(pickled)

    def _wake_locked(self) -> None:
        """Wake callers waiting on the condition and the direct owner."""
        self._cond.notify_all()
        self._owner_wake.set()

    def _mark_taken(self, items: list[_Command]) -> None:
        """Release the admission slots of commands taken off the queue."""
        self._pending_requests -= sum(counted for _, counted, _ in items)

    def _mark_sent(self, items: list[_Command]) -> None:
        """Allow new cancellations for the ids of written cancellations."""
        for _, _, cancellation_id in items:
            if cancellation_id is not None:
                self._pending_cancel_ids.discard(cancellation_id)

    def _check_admission(self) -> None:
        if self._terminal_error is not None:
            raise EngineDeadError(self._terminal_error)
        if self._delivery_error is not None:
            raise EngineDeadError(self._delivery_error)
        if self._closed:
            raise EngineUnavailableError("Executor command sender is stopping")

    def fail(self, error: BaseException) -> None:
        """Stop delivery after an independently confirmed terminal failure.

        Queued commands are dropped. The owner thread then sends the stop
        sentinel if the sender stops workers on failure, and closes the socket.
        """
        with self._cond:
            if self._terminal_error is None:
                self._terminal_error = error
            self._send_sentinel |= self._stop_workers_on_failure
            self._pending.clear()
            self._prepared = None
            self._pending_cancel_ids.clear()
            self._pending_requests = 0
            self._wake_locked()

    def close(
        self,
        timeout: float | None = None,
        *,
        send_sentinel: bool = False,
        cancel_ids: tuple[int, ...] = (),
        wait: bool = True,
    ) -> BaseException | None:
        """Stop admission; the owner thread writes final commands and exits.

        The owner writes the commands already queued, then cancellations for
        ``cancel_ids``, then the stop sentinel, and closes the socket. A
        failure after close() began only ends these writes: it is logged and
        returned, not reported through ``on_failure`` or the error queue.
        During interpreter finalization this returns at once without doing
        anything: the daemon owner thread can no longer run, the ZMQ context
        may be gone, and joining a thread there fails.

        Args:
            timeout: Bound on the final writes and, when waiting, on the join.
            send_sentinel: Also send the worker-stop sentinel.
            cancel_ids: Ids of requests to cancel on rank 0 after the queued
                commands, also when the queue is full. Only the first call's
                ids are used.
            wait: Join the owner thread. Callers on an event loop pass False
                and join later through another close().

        Returns:
            The failure that stopped the sender, or None. Without ``wait``, a
            failure of the final writes is returned by a later close().
        """
        if self._is_finalizing():
            return None
        with self._cond:
            if not self._closed:
                self._shutdown_cancel_ids = cancel_ids
            self._closed = True
            if timeout is not None:
                self._close_deadline = min(self._close_deadline, time.monotonic() + timeout)
            self._send_sentinel |= send_sentinel
            self._wake_locked()
        if wait and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if self._thread.is_alive():
                self._record_failure(TimeoutError("Executor command sender shutdown timed out"))
        with self._cond:
            return self._terminal_error

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
            if lost and self._closed:
                self._peer_gone_while_closing = True
            elif lost and self._delivery_error is None:
                self._delivery_error = ConnectionError(
                    "Executor command connection was lost; request delivery is uncertain"
                )

    def _check_terminal(self) -> None:
        with self._cond:
            self._poll_connection_locked()
            if self._terminal_error is not None:
                raise EngineDeadError(self._terminal_error)
            if self._delivery_error is not None:
                raise self._delivery_error

    def _record_failure(self, error: BaseException) -> None:
        """Fail the sender and report its first failure.

        After close() began, the failure is logged instead, and close()
        returns it: an orderly shutdown must not raise an error that only its
        final writes saw.
        """
        with self._cond:
            report = self._terminal_error is None
            closing = self._closed
        self.fail(error)
        if not report:
            return
        if closing:
            logger.warning(f"Executor command sender could not finish shutdown writes: {error!r}")
            return
        callback = self._on_failure() if self._on_failure is not None else None
        if callback is not None:
            callback(error)
        else:
            self._error_queue.put_nowait(error)

    def _put_until(self, message: bytes, deadline: float = float("inf")) -> None:
        connect_deadline = time.monotonic() + self._connect_timeout
        while True:
            self._check_terminal()
            if self._peer_gone_while_closing:
                raise _PeerGoneWhileClosing()
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
                self._request_queue.put_signed(message, timeout=min(self.POLL_INTERVAL, remaining))
                self._connected = True
                self._check_terminal()
                return
            except zmq.Again:
                with self._cond:
                    self._cond.wait(timeout=min(self.POLL_INTERVAL, remaining))

    def _put_sentinel_after_failure(self) -> None:
        """Queue the stop sentinel after a failure, while rank 0 may still read.

        _put_until cannot be used: it raises once a terminal error is set. The
        send pipe can be full while rank 0 drains large commands, so retry for
        as long as close would linger, unless the connection is lost.
        """
        message = self._encode([_SENTINEL])
        deadline = time.monotonic() + CLOSE_LINGER_MS / 1000
        while True:
            try:
                self._request_queue.put_signed(message, timeout=self.POLL_INTERVAL)
                return
            except zmq.Again:
                if self._monitor is not None:
                    with self._cond:
                        self._poll_connection_locked()
                if self._delivery_error is not None or self._peer_gone_while_closing:
                    raise _PeerGoneWhileClosing()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Executor stop sentinel could not be queued")

    def _take_batch(self) -> list[_Command] | None:
        with self._cond:
            while not self._pending and not self._closed and self._terminal_error is None:
                self._cond.wait(self.POLL_INTERVAL)
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
                if self._peer_gone_while_closing and not self._writing:
                    # Rank 0 is gone, so nothing still queued can arrive.
                    return
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
            # A caller that turns the queue non-empty, closes or fails the
            # sender sets the event. Queued commands are not read here: a
            # caller can hold one while it writes it, and waking for it would
            # only wait for that caller's lock.
            while not self._owner_wake.wait(
                self.POLL_INTERVAL if retry else self.IDLE_CHECK_INTERVAL
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
            else:
                while (batch := self._take_batch()) is not None:
                    self._put_until(self._encode([payload for payload, _, _ in batch]))
                    with self._cond:
                        self._mark_sent(batch)
                        self._cond.notify_all()
        except _PeerGoneWhileClosing:
            # Rank 0 is gone, so nothing still queued can arrive.
            pass
        except Exception as error:
            self._record_failure(error)
        finally:
            self._ready.set()
            with self._cond:
                send_sentinel = self._send_sentinel
            if self._terminal_error is None:
                try:
                    # Attached frontends cancel their own live requests even
                    # when the request queue is full.
                    for start in range(0, len(self._shutdown_cancel_ids), self.MAX_BATCH_SIZE):
                        ids = self._shutdown_cancel_ids[start : start + self.MAX_BATCH_SIZE]
                        self._put_until(
                            self._encode([pickle.dumps(CancellingRequest(i)) for i in ids])
                        )
                except _PeerGoneWhileClosing:
                    # The engine is gone; nothing is left to cancel.
                    pass
                except (zmq.ZMQError, TimeoutError, EngineDeadError) as error:
                    self._record_failure(error)
            sentinel_queued = False
            if send_sentinel:
                try:
                    if self._terminal_error is None:
                        self._put_until(
                            self._encode([_SENTINEL]), time.monotonic() + self.SENTINEL_TIMEOUT
                        )
                    else:
                        self._put_sentinel_after_failure()
                    sentinel_queued = True
                except _PeerGoneWhileClosing:
                    pass
                except (zmq.ZMQError, TimeoutError, EngineDeadError) as error:
                    self._record_failure(error)
            try:
                if self._monitor is not None:
                    try:
                        self._request_queue.socket.disable_monitor()
                    finally:
                        self._monitor.close(linger=0)
            finally:
                # LINGER=0 discards everything still queued in the socket,
                # including a stop sentinel behind large commands. Discard only
                # when the command peer itself is gone, or when a failed
                # sender queued nothing that must still arrive.
                keep_queued = not self._peer_gone_while_closing and (
                    self._terminal_error is None
                    or (sentinel_queued and self._delivery_error is None)
                )
                linger = CLOSE_LINGER_MS if keep_queued else 0
                self._request_queue.socket.setsockopt(zmq.LINGER, linger)
                self._request_queue.close()
