# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The client's state of one open session."""

import asyncio
import threading
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import replace
from typing import Any, Callable, Deque, Dict, List, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm._torch.visual_gen.session_protocol import ResultType, SessionResultMessage
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.session_errors import SessionEnded, SessionError


class PendingSubmit(NamedTuple):
    """The one submit operation in progress: its input and the admission future
    the caller waits on (result: advance index; exception: rejection;
    cancelled: the caller gave up before the reply)."""

    name: str
    admission: Future


def _lost_results_error(error: Optional[str], reason: str, dropped: int) -> str:
    """The terminal error of a session the worker closed for ``reason`` while the
    client still had ``dropped`` results to read; ``error`` is what the terminal
    said before."""
    earlier = [] if error in (None, reason) else [error]
    lost = (
        "1 unread result was discarded"
        if dropped == 1
        else f"{dropped} unread results were discarded"
    )
    return "; ".join(earlier + [reason, lost])


class SessionChannel:
    """Coordinator side of one session: the hand-off point between the main
    thread (the public ``VisualGenSession`` calls) and the client's background
    thread (which dispatches the worker's replies here). Ten members, one per
    fact, each described at its field below.
    """

    def __init__(self, session_id: int):
        self.session_id = session_id
        # The open reply: the capabilities on success, the open error otherwise. Set
        # once, never replaced, so it is read without the lock; its own state is atomic.
        self._opened: Future[Dict[str, Any]] = Future()
        # Guards every member below: which operations exist and that they agree.
        self._lock = threading.Lock()
        # The client's mapping of the delivery slots, slots[slot_id][stream], from a
        # successful open until the session ends; dropping it unmaps the storage.
        self._slots: Optional[List[Dict[str, torch.Tensor]]] = None
        # One CUDA event per slot, for exactly as long as the mapping, when a
        # stream lives in device memory: recorded on the caller's stream as a
        # slot is returned, so the worker's next write waits for the caller's reads.
        self._return_events: Optional[List[torch.cuda.Event]] = None
        # None until a close begins; then the close operation. Completes on the
        # worker's confirmation, or with the sticky failure if the worker group died.
        self._closed: Optional[Future] = None
        # The one submit in flight: input name plus the admission reply. Set until
        # its caller finishes, so a second submit is refused meanwhile.
        self._submit: Optional[PendingSubmit] = None
        # The one read in flight. A fresh future per wait: the dispatcher completes it
        # when the queue changes; the reader re-arms under the lock, never missing one.
        self._reader: Optional[Future] = None
        # Unread stream entries in order: MEDIA and TERMINAL from the worker, then
        # one client-local ENDED or CLOSED entry. Bounded by the slot count plus two.
        self._results: Deque[SessionResultMessage] = deque()
        # Lockstep only: media the worker owes the client and the client has not
        # taken, from the initial result and each admission. advance() refuses while > 0.
        self._lockstep_due = 0

    # ---- derived views

    @property
    def capabilities(self) -> Optional[Dict[str, Any]]:
        fut = self._opened
        if fut.done() and not fut.cancelled() and fut.exception() is None:
            return fut.result()
        return None

    @property
    def slots(self) -> List[Dict[str, torch.Tensor]]:
        """The client's mapping of the delivery slots, ``slots[slot_id][stream]``.
        Only while the session is open. Read from a result builder, which ``take``
        runs inside its critical section, so no lock is taken here."""
        slots = self._slots
        if slots is None:
            raise RuntimeError(f"session {self.session_id} is not open")
        return slots

    def return_event(self, slot_id: int) -> Optional[torch.cuda.Event]:
        """The event to record behind the caller's device reads of slot
        ``slot_id`` before returning it; ``None`` when the session has no
        device-placed stream or is over."""
        events = self._return_events
        return None if events is None else events[slot_id]

    def wait_open(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Block until the worker answered the open; the capabilities, or the open
        error raised."""
        return self._opened.result(timeout)

    def wait_open_async(self) -> "asyncio.Future":
        """The open reply as a future of the running event loop."""
        return asyncio.wrap_future(self._opened)

    def on_open(self, callback: Callable[[Optional[BaseException]], None]) -> None:
        """Run ``callback`` once the open is answered, at once if it already was:
        with ``None`` on success, else with the open error."""

        def done(fut: Future) -> None:
            if fut.cancelled():
                callback(RuntimeError("open cancelled"))
            else:
                callback(fut.exception())

        self._opened.add_done_callback(done)

    @property
    def failure(self) -> Optional[BaseException]:
        """The sticky worker failure, if the worker group is gone."""
        close = self._closed
        if close is not None and close.done() and not close.cancelled():
            return close.exception()
        return None

    @property
    def lease_lost(self) -> Optional[str]:
        """Why views over the slots are no longer usable; ``None`` while they are.
        Set as soon as a close has begun, requested or worker-initiated."""
        close = self._closed
        if close is None:
            return None
        failure = self.failure
        return str(failure) if failure is not None else f"session {self.session_id} is closed"

    def closing(self) -> Optional[Future]:
        """The close operation, if one has begun."""
        return self._closed

    def unread_media(self) -> bool:
        with self._lock:
            return any(m.type is ResultType.MEDIA for m in self._results)

    def lockstep_due(self) -> int:
        with self._lock:
            return self._lockstep_due

    def _queued_input(self) -> Optional[str]:
        caps = self.capabilities or {}
        if not caps.get("lockstep"):
            return None
        for spec in caps.get("inputs", []):
            if spec["policy"] == "queued":
                return spec["name"]
        return None

    # ---- opening (background thread)

    def open_succeeded(
        self,
        capabilities: Dict[str, Any],
        slots: List[Dict[str, torch.Tensor]],
        return_events: Optional[List[torch.cuda.Event]] = None,
    ) -> None:
        with self._lock:
            self._slots = slots
            self._return_events = return_events
            if capabilities.get("lockstep") and capabilities.get("initial_result"):
                self._lockstep_due = 1
        self._complete(self._opened, capabilities)

    def open_failed(self, error: Exception) -> None:
        self._complete(self._opened, error=error)
        with self._lock:
            close = self._closed
        if close is not None:
            self._complete(close, None)  # a close requested while opening: nothing to send

    # ---- results

    def deliver(self, message: SessionResultMessage) -> None:
        with self._lock:
            close = self._closed
            if (close is not None and close.done()) or (
                self._results and self._results[-1].type is ResultType.CLOSED
            ):
                return  # after the end: a stray
            self._results.append(message)
            reader = self._reader
        self._notify(reader)

    def reserve_read(self) -> None:
        with self._lock:
            if self._reader is not None:
                raise SessionError("a result request is already in progress on this session")
            self._reader = Future()

    def release_read(self) -> None:
        with self._lock:
            self._reader = None

    def take(
        self,
        build: Callable[[SessionResultMessage], Any],
        expected_index: Any = None,
    ) -> Any:
        """One step of a reserved read. Returns the built result,
        ``ResultType.ENDED`` when the stream has ended, or ``None`` when nothing
        is readable yet; in that
        case the wakeup is armed in the same critical section, so a delivery,
        close or failure landing afterwards is never missed by the following
        ``notification()``. Raises the sticky failure or ``SessionEnded`` after an
        explicit close. With ``expected_index`` (an advance), media at the head
        carrying another advance index raises ``SessionError`` before anything is
        taken, so the head stays readable. Builds under the lock, so the mapping
        cannot be torn down meanwhile."""
        with self._lock:
            failure = self.failure
            if failure is not None:
                raise failure
            if not self._results:
                if self._closed is not None:
                    raise SessionEnded(f"session {self.session_id} is closed")
                if self._reader is not None and self._reader.done():
                    self._reader = Future()  # re-armed with the predicate still false
                return None
            head = self._results[0]
            if head.type is ResultType.CLOSED:
                raise SessionEnded(f"session {self.session_id} is closed")
            if head.type is ResultType.ENDED:
                return ResultType.ENDED
            if (
                expected_index is not None
                and head.type is ResultType.MEDIA
                and head.advance_index != expected_index
            ):
                raise SessionError(
                    f"advance() admitted index {expected_index} but the next result is advance "
                    f"{head.advance_index}; the worker broke result ordering"
                )
            result = build(head)
            self._results.popleft()
            if head.type is ResultType.TERMINAL:
                self._results.appendleft(
                    SessionResultMessage.end(self.session_id, ResultType.ENDED)
                )
            elif head.type is ResultType.MEDIA and self._lockstep_due > 0:
                self._lockstep_due -= 1
            return result

    def notification(self) -> Future:
        """The future to await before the next ``take``: whatever ``take`` armed
        when it found nothing readable. Never re-armed here: a completed one
        means something changed since, and the next ``take`` sees it."""
        with self._lock:
            if self._reader is None:
                raise SessionError("no read is reserved")
            return self._reader

    def _notify(self, reader: Optional[Future]) -> None:
        if reader is not None:
            self._complete(reader, None)

    # ---- admission

    def reserve_submit(self, name: str, for_advance: bool = False) -> Future:
        """Register the one submit in progress; returns its admission future. For
        an advance, also refuse while an earlier result is still owed; the check
        and the reservation are one step."""
        with self._lock:
            failure = self.failure
            if failure is not None:
                raise failure
            if self._closed is not None:
                raise SessionEnded(f"session {self.session_id} is closed")
            if self._submit is not None:
                raise SessionError("a submit is already in progress on this session")
            if for_advance and self._lockstep_due:
                raise SessionError(
                    f"{self._lockstep_due} earlier result(s) are unread; take them through results() "
                    "before calling advance()"
                )
            fut: Future = Future()
            self._submit = PendingSubmit(name, fut)
            return fut

    def plan_advance(self, name: str) -> Optional[Future]:
        """Start an advance under a reserved read: ``None`` when the head of the
        stream is already a terminal (take it, submit nothing), else the
        admission future of the reserved submit. Sticky failure first; an ended
        or explicitly closed stream ends the call; a retained terminal is
        claimable even after a graceful worker close; only then does a close
        prevent a new submit."""
        with self._lock:
            failure = self.failure
            if failure is not None:
                raise failure
            head = self._results[0] if self._results else None
            if head is not None and head.type in (ResultType.CLOSED, ResultType.ENDED):
                raise SessionEnded(f"session {self.session_id} has ended")
            if head is not None and head.type is ResultType.TERMINAL:
                return None
            if self._closed is not None:
                raise SessionEnded(f"session {self.session_id} is closed")
            if self._submit is not None:
                raise SessionError("a submit is already in progress on this session")
            if self._lockstep_due:
                raise SessionError(
                    f"{self._lockstep_due} earlier result(s) are unread; take them through results() "
                    "before calling advance()"
                )
            fut: Future = Future()
            self._submit = PendingSubmit(name, fut)
            return fut

    def admitted(self, index: Optional[int], error: Optional[Exception]) -> None:
        """The worker's answer to the one pending submit (background thread)."""
        with self._lock:
            pending = self._submit
            if pending is None:
                logger.warning(
                    f"DiffusionClient: admission for session {self.session_id} with no submit pending"
                )
                return
            if error is None and pending.name == self._queued_input():
                self._lockstep_due += 1  # owed from now on, whether or not the caller still waits
            # Completed under the lock: either the caller still owns the future
            # and retires the record when done, or it had cancelled and the
            # record retires here, with the reply. No window between the two.
            if not self._complete(pending.admission, index, error):
                self._submit = None

    def withdraw_submit(self, fut: Future) -> None:
        """A reserved submit that was never sent (refused locally): the record
        goes, nothing is owed, no index moved."""
        with self._lock:
            pending = self._submit
            if pending is not None and pending.admission is fut:
                self._submit = None

    def release_submit(self, fut: Future) -> None:
        """The caller of the submit that owns ``fut`` is done with it. A reply
        already received retires the record; a caller leaving before the reply
        marks it cancelled and the reply retires it."""
        with self._lock:
            pending = self._submit
            if pending is None or pending.admission is not fut:
                return
            if fut.done() and not fut.cancelled():
                self._submit = None
                return
            fut.cancel()

    # ---- close and failure

    def request_close(self) -> Tuple[Future, bool]:
        """The client closes: discard the unread stream, settle the submit in
        progress, expire the leases. Returns the close operation and whether this
        call began it (the caller then arranges the one close command)."""
        with self._lock:
            if self._closed is not None:
                # Already closing or closed (a worker-initiated close may have
                # retained a terminal): the client's close still discards the stream.
                if self.failure is None and not (
                    self._results and self._results[-1].type is ResultType.CLOSED
                ):
                    self._results.clear()
                    self._results.append(
                        SessionResultMessage.end(self.session_id, ResultType.CLOSED)
                    )
                return self._closed, False
            self._closed = Future()
            self._results.clear()
            self._results.append(SessionResultMessage.end(self.session_id, ResultType.CLOSED))
            self._lockstep_due = 0
            pending, self._submit = self._submit, None
            reader = self._reader
        if pending is not None:
            self._complete(pending.admission, error=SessionEnded("session closed"))
        self._notify(reader)
        return self._closed, True

    def worker_closed(self, reason: Optional[str] = None) -> None:
        """The worker confirmed (or initiated) the close. Media still unread is
        dropped with its mappings; a terminal is kept unless the client had
        closed; the close operation completes. When the worker closed on its
        own for ``reason`` and media was dropped, the kept terminal becomes an
        error that says so and how many results were lost."""
        with self._lock:
            if self._closed is None:
                self._closed = Future()
                dropped = sum(1 for m in self._results if m.type is ResultType.MEDIA)
                kept = [
                    m for m in self._results if m.type in (ResultType.ENDED, ResultType.TERMINAL)
                ]
                if dropped and reason is not None:
                    kept = [
                        replace(m, error=_lost_results_error(m.error, reason, dropped))
                        if m.type is ResultType.TERMINAL
                        else m
                        for m in kept
                    ]
                self._results.clear()
                self._results.extend(kept)
            self._lockstep_due = 0
            pending, self._submit = self._submit, None
            reader = self._reader
            self._slots = None
            self._return_events = None
            close = self._closed
        if pending is not None:
            self._complete(pending.admission, error=SessionEnded("session closed"))
        self._complete(close, None)
        self._notify(reader)

    def fail(self, error: Exception) -> None:
        """The worker group is gone. Unless the session already ended gracefully,
        the close operation fails with ``error`` (sticky), every waiter is woken
        with it and the mappings are released. Never raises."""
        with self._lock:
            close = self._closed
            if close is not None and close.done():
                return  # already over
            if close is None:
                self._closed = close = Future()
            # Published inside the critical section: from here every reader and
            # every caller sees the failure, never a graceful close in progress.
            self._complete(close, error=error)
            self._results.clear()
            self._lockstep_due = 0
            pending, self._submit = self._submit, None
            reader = self._reader
            self._slots = None
            self._return_events = None
        # The opening completes after the close outcome: a close callback that
        # fires on it finds the failure already recorded and cannot soften it.
        self._complete(self._opened, error=error)
        if pending is not None:
            self._complete(pending.admission, error=error)
        self._notify(reader)

    @staticmethod
    def _complete(fut: Future, value: Any = None, error: Optional[BaseException] = None) -> bool:
        """Complete a future the other side may have cancelled or completed."""
        try:
            if error is not None:
                fut.set_exception(error)
            else:
                fut.set_result(value)
            return True
        except InvalidStateError:
            return False
