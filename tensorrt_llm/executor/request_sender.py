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
    """

    def __init__(self) -> None:
        self._handshaking: set[int] = set()
        # Empty until a handshake completes; None if no candidate was observed.
        self._peer: set[int] | None = set()
        self._handshake_failed = False

    def observe(self, event: int, value: int) -> bool:
        """Record one monitor event; return whether the command peer was lost."""
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
                if handshake_failed and len(self._peer) > 1:
                    self._peer.discard(value)
                    return False
                return True
        return False


class _RequestSender:
    """Own command sends and closure, including the worker shutdown message.

    Connected backpressure retains queued commands without declaring the engine
    dead. Losing an established direct connection makes delivery uncertain, so
    pending requests fail rather than replaying commands after reconnection.

    With ``direct_send``, a caller writes queued commands itself while the owner
    thread is idle. CPython can deny a waiting thread the GIL for as long as
    another thread keeps briefly releasing it, as a busy asyncio loop does, so
    unbatched admission must not depend on scheduling the owner thread. The
    condition lock serializes every socket use, which provides the memory
    barrier libzmq requires when a socket changes threads. The owner still
    handles backpressure, connection setup and shutdown.
    """

    MAX_BATCH_SIZE = 1024
    MAX_PENDING_REQUESTS = 1024
    MAX_PENDING_CONTROLS = 1024
    MAX_INFLIGHT_BATCHES = _COMMAND_QUEUE_HWM
    SEND_TIMEOUT = 0.05
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
        self._connection_lost = False
        self._connected = False
        self._direct_send = direct_send and not batching
        # The owner thread has taken a batch and may be writing the socket.
        self._sending = False
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
            return self._terminal_error is None and not self._closed

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
                self._cond.wait(self.SEND_TIMEOUT if self._direct_send else None)
                self._check_admission()
            self._pending.append((payload, control, cancellation_id))
            if cancellation_id is not None:
                self._pending_cancel_ids.add(cancellation_id)
            elif control:
                self._pending_controls += 1
            else:
                self._pending_requests += 1
            self._write_pending_locked()
            self._cond.notify_all()

    def _write_pending_locked(self) -> None:
        """Write queued commands on the calling thread while the owner is idle."""
        if (
            not self._direct_send
            or self._sending
            or self._closed
            or self._terminal_error is not None
        ):
            return
        self._poll_connection_locked()
        if self._connection_lost:
            # The owner fails accepted work; nothing may follow a lost peer.
            return
        while self._pending:
            item = self._pending[0]
            try:
                self._request_queue.put([item[0]], timeout=0)
            except zmq.Again:
                # Backpressure and connection setup belong to the owner.
                return
            except Exception:
                # Leave the command to the owner, which records the failure.
                self._direct_send = False
                return
            self._pending.popleft()
            self._connected = True
            self._mark_taken((item,))
            self._mark_sent((item,))

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
            self._cond.notify_all()

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
            self._cond.notify_all()
        if thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                self._record_failure(TimeoutError("Executor command sender shutdown timed out"))

    def _poll_connection_locked(self) -> None:
        while self._monitor.poll(0):
            message = recv_monitor_message(self._monitor)
            event = message["event"]
            if event == zmq.EVENT_HANDSHAKE_SUCCEEDED:
                self._connected = True
            if self._connection.observe(event, message["value"]) and not self._closed:
                self._connection_lost = True

    def _check_terminal(self) -> None:
        with self._cond:
            self._poll_connection_locked()
            if self._terminal_error is not None:
                raise EngineDeadError(self._terminal_error)
            if self._connection_lost:
                raise ConnectionError(
                    "Executor command connection was lost; request delivery is uncertain"
                )

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
            self._sending = True
            self._cond.notify_all()
            return items

    def _run(self) -> None:
        try:
            # Connect-side sockets must not report successful delivery by
            # buffering commands for a peer that has not connected yet.
            self._request_queue.socket.setsockopt(zmq.IMMEDIATE, 1)
            self._monitor = self._request_queue.socket.get_monitor_socket(events=_MONITOR_EVENTS)
            self._request_queue.setup_lazily()
            self._ready.set()
            while (batch := self._take_batch()) is not None:
                self._put_until([payload for payload, _, _ in batch])
                with self._cond:
                    self._mark_sent(batch)
                    self._sending = False
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
