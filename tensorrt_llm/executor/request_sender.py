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

"""Single-owner command delivery and bounded worker-connection recovery."""

import collections

# Only serializes executor commands; this owner does not deserialize input.
import pickle  # nosec B403
import sys
import threading
import time
import weakref
from queue import Queue

import zmq

from .ipc import IpcQueue
from .request import _COMMAND_QUEUE_HWM, CancellingRequest, _CommandBarrier
from .utils import EngineDeadError, EngineUnavailableError


class _RequestSender:
    """Own command sends and closure, including the worker shutdown message.

    Ordinary queue saturation rejects admission. Cancellation retains one
    intent per admitted request until delivery. A connection outage pauses
    admission until a worker barrier confirms receipt, or one absolute recovery
    deadline expires.
    """

    MAX_BATCH_SIZE = 1024
    MAX_PENDING_REQUESTS = 1024
    MAX_PENDING_CONTROLS = 1024
    MAX_INFLIGHT_BATCHES = _COMMAND_QUEUE_HWM
    SEND_TIMEOUT = 0.05
    RECOVERY_TIMEOUT = 5.0
    ACK_RETRY_INTERVAL = 0.25

    def __init__(
        self,
        request_queue: IpcQueue,
        error_queue: Queue,
        *,
        batching: bool = True,
        on_failure: weakref.WeakMethod | None = None,
        recovery_timeout: float = RECOVERY_TIMEOUT,
    ) -> None:
        self._request_queue = request_queue
        self._error_queue = error_queue
        self._batching = batching
        self._on_failure = on_failure
        self._recovery_timeout = recovery_timeout
        self._pending = collections.deque()
        self._pending_requests = 0
        self._pending_controls = 0
        self._pending_cancel_ids: set[int] = set()
        self._enqueued_count = 0
        self._sent_count = 0
        self._cond = threading.Condition()
        self._closed = False
        self._send_sentinel = False
        self._shutdown_cancel_ids: tuple[int, ...] = ()
        self._stopping = False
        self._recovering = False
        self._terminal_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._is_finalizing = sys.is_finalizing
        self._ack_queue: IpcQueue | None = None
        self._probe_sequence = 0
        self._control_deadline = 0.0
        self._close_deadline = float("inf")

    @property
    def available(self) -> bool:
        """Whether normal admission is currently possible."""
        with self._cond:
            return not self._recovering and self._terminal_error is None and not self._closed

    @property
    def recovering(self) -> bool:
        """Whether new admission is paused for connection recovery."""
        with self._cond:
            return self._recovering

    def _start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="proxy_request_sender"
            )
            self._thread.start()

    def send(
        self, payload: bytes, *, control: bool = False, cancellation_id: int | None = None
    ) -> None:
        """Queue a captured command, rejecting work that cannot be admitted."""
        with self._cond:
            if self._terminal_error is not None:
                raise EngineDeadError(self._terminal_error)
            if self._closed:
                raise EngineUnavailableError("Executor command sender is stopping")
            if cancellation_id is not None and cancellation_id in self._pending_cancel_ids:
                return
            if not control and cancellation_id is None and self._recovering:
                raise EngineUnavailableError("Executor command connection is recovering")
            count = self._pending_controls if control else self._pending_requests
            limit = self.MAX_PENDING_CONTROLS if control else self.MAX_PENDING_REQUESTS
            if cancellation_id is None and count >= limit:
                raise EngineUnavailableError("Executor command queue is full")
            self._pending.append((payload, control, cancellation_id))
            self._enqueued_count += 1
            ticket = self._enqueued_count
            if cancellation_id is not None:
                self._pending_cancel_ids.add(cancellation_id)
            elif control:
                self._pending_controls += 1
            else:
                self._pending_requests += 1
            self._start()
            self._cond.notify_all()
            if not self._batching:
                # Release the GIL until this command is written or recovery
                # starts, before the caller resumes Python preprocessing work.
                self._cond.wait_for(
                    lambda: self._sent_count >= ticket
                    or self._recovering
                    or self._terminal_error is not None
                )
                if self._terminal_error is not None:
                    raise EngineDeadError(self._terminal_error)

    def fail(self, error: BaseException) -> None:
        """Stop delivery after an independently confirmed terminal failure."""
        with self._cond:
            if self._terminal_error is None:
                self._terminal_error = error
            self._pending.clear()
            self._pending_cancel_ids.clear()
            self._pending_controls = self._pending_requests = 0
            self._cond.notify_all()

    def set_control_deadline(self, deadline: float) -> None:
        """Account for an in-progress synchronous worker control operation."""
        with self._cond:
            self._control_deadline = deadline

    def close(
        self,
        timeout: float | None = None,
        *,
        send_sentinel: bool = False,
        cancel_ids: tuple[int, ...] = (),
    ) -> bool:
        """Request bounded owner-thread shutdown; callers never write the socket."""
        if self._is_finalizing():
            return False
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
        return False

    def _check_terminal(self) -> None:
        with self._cond:
            if self._terminal_error is not None:
                raise EngineDeadError(self._terminal_error)

    def _record_failure(self, error: BaseException) -> None:
        with self._cond:
            report = self._terminal_error is None
        self.fail(error)
        callback = self._on_failure() if self._on_failure is not None else None
        if report and callback is not None:
            callback(error)
        elif report:
            self._error_queue.put_nowait(error)

    def _put_until(self, message: object, deadline: float) -> None:
        while True:
            self._check_terminal()
            remaining = min(deadline, self._close_deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Executor command connection recovery timed out")
            try:
                self._request_queue.put(message, timeout=min(self.SEND_TIMEOUT, remaining))
                return
            except zmq.Again:
                with self._cond:
                    self._recovering = True
                    self._cond.notify_all()
                    self._cond.wait(timeout=min(self.SEND_TIMEOUT, remaining))

    def _confirm_recovery(self, deadline: float) -> None:
        if self._ack_queue is None:
            self._ack_queue = IpcQueue(
                is_server=True, socket_type=zmq.PULL, name="proxy_command_ack"
            )
            self._ack_queue.socket.setsockopt(zmq.LINGER, 0)
        self._probe_sequence += 1
        token = self._probe_sequence
        probe = _CommandBarrier(token, self._ack_queue.address)
        payload = [pickle.dumps(probe)]
        self._put_until(payload, deadline)
        retry_at = time.monotonic() + self.ACK_RETRY_INTERVAL
        while True:
            self._check_terminal()
            remaining = min(deadline, self._close_deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Worker did not acknowledge command recovery")
            if self._ack_queue.poll(min(self.SEND_TIMEOUT, remaining)):
                if self._ack_queue.get() == token:
                    with self._cond:
                        self._recovering = False
                        self._cond.notify_all()
                    return
            if time.monotonic() >= retry_at:
                # A worker can lose its short-lived ACK connection. Retrying
                # the barrier is safe; generation commands must not be replayed.
                self._put_until(payload, deadline)
                retry_at = time.monotonic() + self.ACK_RETRY_INTERVAL

    def _take_batch(self) -> list[tuple[bytes, bool, int | None]] | None:
        with self._cond:
            while not self._pending and not self._closed and self._terminal_error is None:
                self._cond.wait()
            if self._terminal_error is not None:
                raise EngineDeadError(self._terminal_error)
            if not self._pending:
                return None
            size = self.MAX_BATCH_SIZE if self._batching else 1
            items = [self._pending.popleft() for _ in range(min(size, len(self._pending)))]
            for _, control, cancellation_id in items:
                if cancellation_id is not None:
                    continue
                elif control:
                    self._pending_controls -= 1
                else:
                    self._pending_requests -= 1
            return items

    def _run(self) -> None:
        try:
            # Connect-side sockets must not report successful delivery by
            # buffering commands for a peer that has not connected yet.
            self._request_queue.socket.setsockopt(zmq.IMMEDIATE, 1)
            self._request_queue.socket.setsockopt(zmq.SNDHWM, self.MAX_INFLIGHT_BATCHES)
            while (batch := self._take_batch()) is not None:
                with self._cond:
                    deadline = max(
                        time.monotonic() + self._recovery_timeout, self._control_deadline
                    )
                self._put_until([payload for payload, _, _ in batch], deadline)
                with self._cond:
                    self._sent_count += len(batch)
                    for _, _, cancellation_id in batch:
                        if cancellation_id is not None:
                            self._pending_cancel_ids.discard(cancellation_id)
                    recovering = self._recovering
                    self._cond.notify_all()
                if recovering:
                    self._confirm_recovery(deadline)
        except Exception as error:
            self._record_failure(error)
            # A failure consumer may request an orderly stop of surviving
            # workers. Keep that final request on the socket owner as well.
            with self._cond:
                self._cond.wait_for(lambda: self._closed, timeout=1.0)
        finally:
            with self._cond:
                self._stopping = True
                send_sentinel = self._send_sentinel
            if self._terminal_error is None:
                try:
                    # Attached frontends cancel their own live requests even
                    # when ordinary control capacity was already exhausted.
                    for start in range(0, len(self._shutdown_cancel_ids), self.MAX_BATCH_SIZE):
                        ids = self._shutdown_cancel_ids[start : start + self.MAX_BATCH_SIZE]
                        self._put_until(
                            [pickle.dumps(CancellingRequest(i)) for i in ids],
                            time.monotonic() + self._recovery_timeout,
                        )
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
            if self._ack_queue is not None:
                self._ack_queue.close()
            linger = 0 if self._terminal_error is not None else 1000
            self._request_queue.socket.setsockopt(zmq.LINGER, linger)
            self._request_queue.close()
