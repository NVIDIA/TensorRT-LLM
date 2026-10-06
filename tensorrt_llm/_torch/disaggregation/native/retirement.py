# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deadline arbitration that never queries or releases a transfer backend."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal

from tensorrt_llm import logger


@dataclass(frozen=True)
class QuiescenceFatalEvent:
    """Immutable evidence that an exposed owner missed its retirement deadline."""

    request_id: int
    direction: Literal["send", "receive"]
    reason: str
    started_at: float
    deadline: float
    expired_at: float


class RetirementDeadline:
    """Arbitrate one session's physical claims against a non-resettable deadline."""

    def __init__(
        self,
        controller: RetirementWatchdog,
        request_id: int,
        direction: Literal["send", "receive"],
        timeout_s: float,
    ) -> None:
        """Initialize an unpublished owner without starting either clock.

        Args:
            controller: Shared, backend-free watchdog and admission arbiter.
            request_id: Request identity within this worker.
            direction: Whether this session owns source or destination memory.
            timeout_s: Request timeout and subsequent quiescence grace duration.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("retirement requires a finite positive transfer timeout")
        self.controller = controller
        self.request_id = request_id
        self.direction = direction
        self.timeout_s = timeout_s
        self._request_deadline: float | None = None
        self._drain_started: float | None = None
        self._reason = ""
        self._claims: dict[int, object] = {}
        self._pieces_complete = False
        self._closed = False
        self._timeout_outcome: Callable[[], None] | None = None

    def bind_timeout_outcome(self, callback: Callable[[], None]) -> None:
        """Bind the session's metadata-only logical timeout transition.

        Args:
            callback: Updates only logical outcome metadata under the shared lock.
                It must not call a backend, acquire a session lock, or perform I/O.
        """
        with self.lock:
            self._timeout_outcome = callback

    def check(self) -> None:
        """Commit any elapsed timeout before a logical or physical transition."""
        with self.lock:
            self._check_locked(self.controller.clock())

    @property
    def lock(self) -> threading.RLock:
        """Return the arbiter shared by exposure, physical settlement and expiry."""
        return self.controller.lock

    def expose(self, *owners: object) -> bool:
        """Atomically retain owners before publication or submission can escape.

        Args:
            owners: Physical operations or receive owners to keep alive together.

        Returns:
            Whether this operation may cross the publication/submission boundary.
        """
        with self.lock:
            now = self.controller.clock()
            self._check_locked(now)
            if (
                self._closed
                or self._pieces_complete
                or self._drain_started is not None
                or self.controller.fatal is not None
            ):
                return False
            if self._request_deadline is None:
                self._request_deadline = now + self.timeout_s
            self._claims.update((id(owner), owner) for owner in owners)
            self.controller.wake.set()
            return True

    def complete_pieces(self) -> None:
        """Seal successful delivery of every expected piece, including required AUX.

        Physical claims remain independently monitored until safe evidence settles
        them; successful logical delivery alone cannot disable the deadline.
        """
        with self.lock:
            self._check_locked(self.controller.clock())
            self._pieces_complete = True

    @property
    def is_complete(self) -> bool:
        """Return successful whole-session closure, not merely an idle claim set."""
        with self.lock:
            self._check_locked(self.controller.clock())
            return (
                self._pieces_complete
                and not self._claims
                and self._drain_started is None
                and self.controller.fatal is None
            )

    def request_drain(self, reason: str) -> None:
        """Record the first terminal trigger; later triggers cannot extend grace.

        Args:
            reason: Diagnostic explanation of cancellation, failure or shutdown.
        """
        with self.lock:
            now = self.controller.clock()
            self._check_locked(now)
            if self._closed or self._drain_started is not None:
                return
            self._drain_started = now
            self._reason = reason
            self.controller.wake.set()

    def settle(self, owner: object) -> bool:
        """Authorize retirement only if safe evidence wins before fatal expiry.

        Args:
            owner: The exact owner previously exposed by this session.

        Returns:
            Whether the caller may release that owner's physical roots.
        """
        with self.lock:
            self._check_locked(self.controller.clock())
            if self.controller.fatal is not None:
                return False
            self._claims.pop(id(owner), None)
            return True

    def retain_unproven(self, owner: object, reason: str) -> None:
        """Restore disputed evidence to deadline tracking without reopening admission.

        Args:
            owner: An existing physical owner whose safe evidence is now disputed.
            reason: Diagnostic explanation for the first ambiguity trigger.
        """
        with self.lock:
            if self._closed:
                return
            # A successfully completed session has no running request clock. Start grace
            # from this new ambiguity, but preserve any earlier drain trigger.
            self.request_drain(reason)
            self._claims[id(owner)] = owner
            self._check_locked(self.controller.clock())
            self.controller.wake.set()

    def can_retire(self) -> bool:
        """Return whether no physical claim remains and fatal expiry has not won."""
        with self.lock:
            self._check_locked(self.controller.clock())
            return not self._claims and self.controller.fatal is None

    def close(self) -> bool:
        """Idempotently remove a safely retired session from watchdog tracking."""
        with self.lock:
            if not self.can_retire():
                return False
            self._closed = True
            self.controller._owners.discard(self)
            return True

    def _check_locked(self, now: float) -> None:
        """Latch expiry using metadata only while holding the shared arbiter.

        Args:
            now: Monotonic clock sample used for the expiry decision.
        """
        if self._closed or self.controller.fatal is not None:
            return
        session_complete = self._pieces_complete and not self._claims
        if (
            not session_complete
            and self._drain_started is None
            and self._request_deadline is not None
        ):
            if now >= self._request_deadline:
                self._drain_started = self._request_deadline
                self._reason = "transfer timeout"
                if self._timeout_outcome is not None:
                    self._timeout_outcome()
        if self._claims and self._drain_started is not None:
            deadline = self._drain_started + self.timeout_s
            if now >= deadline:
                self.controller.fatal = QuiescenceFatalEvent(
                    self.request_id,
                    self.direction,
                    self._reason,
                    self._drain_started,
                    deadline,
                    now,
                )
                self.controller.wake.set()


class RetirementWatchdog:
    """Progress deadlines independently from backend waits, queries and executor polling."""

    def __init__(
        self,
        callback: Callable[[QuiescenceFatalEvent], None],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a stopped watchdog with a qualified containment callback.

        Args:
            callback: Fatal notification; must not perform ordinary resource cleanup.
            clock: Monotonic clock, injectable for deterministic deadline tests.
        """
        self.lock = threading.RLock()
        self.clock = clock
        self.wake = threading.Event()
        self.fatal: QuiescenceFatalEvent | None = None
        self._callback = callback
        self._notified = False
        self._owners: set[RetirementDeadline] = set()
        self._stopped = threading.Event()
        self._admission_closed = False
        self._thread: threading.Thread | None = None

    def create_owner(
        self, request_id: int, direction: Literal["send", "receive"], timeout_s: float
    ) -> RetirementDeadline:
        """Register one session before it becomes visible to transfer workers.

        Args:
            request_id: Session's request identity.
            direction: Source or destination ownership role.
            timeout_s: Finite positive request and quiescence timeout.

        Returns:
            The session's ownership/deadline arbiter.

        Raises:
            RuntimeError: Admission is closed following fatal expiry or shutdown.
        """
        with self.lock:
            if self.fatal is not None or self._admission_closed or self._stopped.is_set():
                raise RuntimeError("KV retirement admission is closed")
            owner = RetirementDeadline(self, request_id, direction, timeout_s)
            self._owners.add(owner)
            return owner

    def start(self) -> None:
        """Start the single backend-free progress thread idempotently."""
        with self.lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="kv-retirement", daemon=True)
            self._thread.start()

    def progress(self) -> None:
        """Evaluate all clocks, then deliver at most one fatal event outside locks."""
        notify = None
        with self.lock:
            now = self.clock()
            for owner in self._owners:
                owner._check_locked(now)
            if self.fatal is not None and not self._notified:
                self._notified = True
                notify = self.fatal
        if notify is not None:
            try:
                self._callback(notify)
            except Exception as error:
                # The containment boundary is external. Failure cannot erase its
                # fatal decision or destroy the owners that still retain memory.
                logger.error(f"KV retirement containment callback failed; owners retained: {error}")

    def request_shutdown(self) -> None:
        """Start drain on every tracked owner before any resource teardown."""
        with self.lock:
            self._admission_closed = True
            for owner in self._owners:
                owner.request_drain("shutdown")

    @property
    def admission_closed(self) -> bool:
        """Return the irreversible shutdown gate, without entering any session lock."""
        with self.lock:
            return self._admission_closed

    def stop(self) -> None:
        """Stop progress only after every owner has safely closed.

        Raises:
            RuntimeError: Ownership remains tracked, including fatal quarantine.
        """
        with self.lock:
            if self._owners or self.fatal is not None:
                raise RuntimeError("cannot stop KV retirement with retained owners")
            self._stopped.set()
            self.wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def require_retired(self) -> None:
        """Refuse ordinary teardown while any session or fatal quarantine is retained.

        Raises:
            RuntimeError: A session still owns resources or fatal expiry won.
        """
        with self.lock:
            if self._owners or self.fatal is not None:
                raise RuntimeError("KV retirement still retains resources; teardown refused")

    def _run(self) -> None:
        """Poll metadata without acquiring a backend, CUDA, or collective lock."""
        while not self._stopped.is_set():
            self.wake.wait(0.01)
            self.wake.clear()
            self.progress()
            if self.fatal is not None:
                # The bound callback roots the worker, sessions, pools and agent.
                # If containment returns/raises, retain those roots until process
                # teardown; stop() deliberately cannot release a fatal watchdog.
                self._stopped.wait()
                return
