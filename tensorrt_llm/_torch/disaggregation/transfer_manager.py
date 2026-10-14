# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-free ownership of acquired leases, backend Attempts and safe retirement."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from tensorrt_llm.logger import logger

from ._assembly import (
    BackendFactory,
    FactoryRejected,
    ProviderCleanupCompleteFn,
    RegionNamespace,
    project_view,
)
from .base.backend import Attempt, Backend, Hint, Outcome, SubmissionRejected, WakeFn
from .lifecycle.retirement import RetirementDeadline, RetirementWatchdog

if TYPE_CHECKING:
    from ..pyexecutor.kv_cache.sharing import Lease, LeaseView


class Direction(Enum):
    """The lending-call provenance supplied by orchestration."""

    FETCH = "fetch"
    PUBLISH = "publish"


@dataclass(frozen=True)
class TransferIntent:
    """Immutable operation input; the caller must also freeze its opaque hint.

    Direction comes from the lending call, not introspection of the Lease. Memory
    kind must match independently verified assembly metadata.
    """

    binding_id: str
    profile_id: str
    direction: Direction
    memory_kind: Literal["HOST", "DEVICE"]
    hint: Hint = None


@dataclass(frozen=True)
class Accepted:
    """Ownership transferred permanently to the holder before any exposure."""

    operation_id: int


@dataclass(frozen=True)
class Deferred:
    """Admission is closed; the caller still owns the untouched lease."""

    reason: str


@dataclass(frozen=True)
class Rejected:
    """The intent is invalid; the caller still owns the untouched lease."""

    reason: str


SubmitResult = Accepted | Deferred | Rejected


class Marking(Enum):
    """One-shot write-consumption decision; read leases remain UNDECIDED."""

    UNDECIDED = "undecided"
    DELIVERED = "delivered"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class OperationObservation:
    """Operation facts only; orchestration owns readiness and request recovery."""

    operation_id: int
    outcome: Outcome | None
    fault: str | None
    marking: Marking
    access_ended: bool
    retired: bool


@dataclass(frozen=True)
class LifecycleBinding:
    """An implemented Lifecycle policy bound to one verified assembly profile.

    Only the conservative adapter exists today. Scoped confinement requires the
    Lifecycle follow-up and provider evidence; HOST memory alone cannot select it.
    """

    watchdog: RetirementWatchdog
    timeout_s: float
    profile_id: str = "executor-fatal"

    def __post_init__(self) -> None:
        """Reject unimplemented policy selection or invalid safety timing."""
        if self.profile_id != "executor-fatal":
            raise ValueError("Lifecycle scoped binding is not implemented")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Lifecycle timeout must be finite and positive")


class Submission(Enum):
    """Disposition of the single backend call; uncertainty never permits retry."""

    NOT_CALLED = "not_called"
    RETURNED = "returned"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class _OwnerCleanup:
    """Bounded thread-safe dispatch of provider-owned manager-thread finalizers."""

    def __init__(self, wake: WakeFn, capacity: int = 128) -> None:
        """Retain accepted callbacks until owner-loop execution.

        Args:
            wake: Nonblocking owner-loop notification.
            capacity: Maximum accepted callbacks awaiting execution.
        """
        self._wake = wake
        self._capacity = capacity
        self._lock = threading.Lock()
        self._queue: deque[Callable[[], None]] = deque()
        self._failed: list[Callable[[], None]] = []
        self.closed = False
        self.fault: str | None = None

    def post(self, finalize: Callable[[], None]) -> bool:
        """Accept exactly one execution or leave ownership with the provider.

        Args:
            finalize: Bounded manager-thread finalizer; never blocking SDK cleanup.

        Returns:
            True only if the dispatcher retained the callback for execution.
        """
        with self._lock:
            if self.closed or len(self._queue) >= self._capacity:
                return False
            self._queue.append(finalize)
        self._wake()
        return True

    def drain(self, limit: int = 32) -> None:
        """Execute a bounded batch once, retaining a fault after any exception.

        Args:
            limit: Maximum callbacks executed in this owner-loop round.
        """
        for _ in range(limit):
            with self._lock:
                if not self._queue:
                    return
                finalize = self._queue.popleft()
            try:
                finalize()
            except Exception as error:
                # A failed finalizer may have partly run, so it must not be retried.
                self._failed.append(finalize)
                self.fault = f"owner cleanup failed: {error}"

    def finish(self) -> bool:
        """Close an empty fault-free queue after independent provider cleanup evidence.

        Returns:
            Whether all accepted owner callbacks finished without a known fault.
        """
        with self._lock:
            if self._queue or self.fault is not None:
                return False
            self.closed = True
            return True

    def pending(self) -> bool:
        """Return whether accepted manager-thread finalizers still need a round."""
        with self._lock:
            return bool(self._queue)


@dataclass
class BackendBinding:
    """A provider root with descriptors and cleanup, never a manager/lender service."""

    namespace: RegionNamespace
    profile_id: str
    lifecycle_binding: LifecycleBinding
    factory: BackendFactory
    cleanup: _OwnerCleanup
    provider_cleanup_complete: ProviderCleanupCompleteFn
    backend: Backend | None = None
    accepting: bool = True
    construction_uncertain: bool = False
    retained_labels: frozenset | None = None
    finalized: bool = False
    shutdown_started: bool = False
    shutdown_deadline: float | None = None
    shutdown_thread: threading.Thread | None = None
    shutdown_results: deque[tuple[frozenset | None, str | None]] = field(default_factory=deque)
    shutdown_lock: threading.Lock = field(default_factory=threading.Lock)
    shutdown_stop: threading.Event = field(default_factory=threading.Event)
    lifecycle: RetirementDeadline | None = None
    fault: str | None = None


@dataclass
class OperationRecord:
    """One accepted lease and one fixed-source Attempt, independent of requests."""

    id: int
    intent: TransferIntent
    lease: Lease
    accepted_at: float
    lease_view: LeaseView | None = None
    attempt: Attempt | None = None
    submission: Submission = Submission.NOT_CALLED
    outcome: Outcome | None = None
    marking: Marking = Marking.UNDECIDED
    mark_attempted: bool = False
    released: bool = False
    release_attempted: bool = False
    cancel_sent: bool = False
    abandoned: bool = False
    access_ended: bool = False
    unsafe_lease: bool = False
    exposed_at: float | None = None
    lifecycle: RetirementDeadline | None = None
    timeout: threading.Event = field(default_factory=threading.Event)
    fault: str | None = None
    observed: tuple | None = None


@dataclass(frozen=True)
class ShutdownReport:
    """Joint completion of results, leases, provider cleanup and Lifecycle roots."""

    complete: bool
    retained_bindings: tuple[str, ...]


class TransferManager:
    """Hold operations on one manager thread while backend I/O progresses independently.

    There is no production activation here: request admission, loop-head lending,
    readiness and fallback belong to orchestration. Scoped Lifecycle qualification
    and general lender memory/row capabilities remain activation prerequisites.
    """

    def __init__(self, *, wake: WakeFn, clock: Callable[[], float] = time.monotonic) -> None:
        """Create an empty operation holder.

        Args:
            wake: Thread-safe, nonblocking notification; it must never raise.
            clock: Monotonic observation clock, shared with bound Lifecycle clocks.
        """
        self.bindings: dict[str, BackendBinding] = {}
        self.operations: dict[int, OperationRecord] = {}
        self._wake = wake
        self._clock = clock
        self._owner_thread: int | None = None
        self._builder_thread = threading.get_ident()
        self._operation_id = 0
        self._shutdown_deadline: float | None = None
        self._progress_wake = threading.Event()
        self.counters: dict[str, int] = {}

    def _count(self, key: str) -> None:
        """Increment one payload-free owner-thread operation counter."""
        self.counters[key] = self.counters.get(key, 0) + 1

    def notify(self) -> None:
        """Wake idle holder progress without executing I/O or publishing evidence."""
        self._progress_wake.set()
        self._wake()

    def wait_for_progress(self, timeout: float = 0.01) -> None:
        """Wait for a backend wake or bounded heartbeat without gating collectives.

        Args:
            timeout: Maximum idle delay in seconds.
        """
        self._progress_wake.wait(timeout)
        self._progress_wake.clear()

    @property
    def shutdown_timeout_s(self) -> float:
        """Return the largest bound used by attached Lifecycle profiles."""
        return max((item.lifecycle_binding.timeout_s for item in self.bindings.values()), default=0)

    def _owner(self) -> None:
        """Bind runtime ownership once and reject calls on another manager thread."""
        identity = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = identity
        elif self._owner_thread != identity:
            raise RuntimeError("TransferManager requires its manager owner thread")

    def assemble(
        self,
        binding_id: str,
        namespace: RegionNamespace,
        profile_id: str,
        factory: BackendFactory,
        *,
        lifecycle: LifecycleBinding,
        provider_cleanup_complete: ProviderCleanupCompleteFn,
        config: object = None,
    ) -> None:
        """Root provider construction ownership before invoking its private factory.

        Args:
            binding_id: Unique fixed provider binding identifier.
            namespace: Verified complete regions and assembly-known memory kind.
            profile_id: Exact compatible assembly profile.
            factory: Private construction owner retaining partial resources on failure.
            lifecycle: Implemented policy selected explicitly for this profile.
            provider_cleanup_complete: Thread-safe nonblocking private owner query.
                Monotonic True means no further finalizers will be posted and no
                owner-cleanup obligations remain outside the accepted queue. It
                proves neither ended access nor completion of accepted callbacks.
            config: Opaque provider configuration.

        Raises:
            ValueError: Binding/profile identity is empty or already registered.
            RuntimeError: Assembly runs on the wrong thread or after shutdown.
            FactoryRejected: Constructor certifies no retained resources or access.
            Exception: An uncertain constructor failure; its owner remains retained.
        """
        expected = self._builder_thread if self._owner_thread is None else self._owner_thread
        if threading.get_ident() != expected or self._shutdown_deadline is not None:
            raise RuntimeError("backend assembly is unavailable on this thread or after shutdown")
        if not binding_id or binding_id in self.bindings or not profile_id:
            raise ValueError("binding and profile identifiers must be nonempty and unique")
        cleanup = _OwnerCleanup(self.notify)
        binding = BackendBinding(
            namespace, profile_id, lifecycle, factory, cleanup, provider_cleanup_complete
        )
        self.bindings[binding_id] = binding
        try:
            binding.backend = factory(namespace.regions, self.notify, config, cleanup.post)
            if binding.backend is None:
                raise TypeError("backend factory returned no Backend")
        except FactoryRejected:
            # Certified construction rejection cannot leave accepted cleanup behind.
            if cleanup.pending():
                binding.construction_uncertain = True
                self._stop_binding(binding_id, "rejected constructor left owner cleanup")
            else:
                cleanup.finish()
                del self.bindings[binding_id]
            raise
        except Exception as error:
            binding.construction_uncertain = True
            self._stop_binding(binding_id, f"backend construction failed: {error}")
            raise

    def submit(self, lease: Lease, intent: TransferIntent) -> SubmitResult:
        """Accept ownership before polling or exposing an acquired lease.

        Args:
            lease: Caller-owned read or write lease, possibly still pending.
            intent: Frozen fixed-binding intent consistent with its lending call.

        Returns:
            Accepted transfers ownership permanently. Deferred/Rejected leave the
            lease untouched and caller-owned, without any backend exposure.
        """
        self._owner()
        binding = self.bindings.get(intent.binding_id)
        if binding is None or binding.backend is None:
            return Rejected("unknown or unavailable backend binding")
        if (
            intent.profile_id != binding.profile_id
            or intent.memory_kind != binding.namespace.memory_kind
        ):
            return Rejected("intent does not match verified assembly metadata")
        if intent.direction not in (Direction.FETCH, Direction.PUBLISH):
            return Rejected("unsupported transfer direction")
        if any(record.lease is lease for record in self.operations.values()):
            return Rejected("lease ownership already transferred")
        watchdog = binding.lifecycle_binding.watchdog
        if (
            self._shutdown_deadline is not None
            or not binding.accepting
            or watchdog.admission_closed
            or watchdog.fatal is not None
        ):
            return Deferred("binding admission is closed")
        caps = binding.backend.capabilities
        if not (caps.fetch if intent.direction is Direction.FETCH else caps.publish):
            return Rejected("backend does not support the requested direction")
        self._operation_id += 1
        record = OperationRecord(self._operation_id, intent, lease, self._clock())
        self.operations[record.id] = record
        self._count("accepted")
        return Accepted(record.id)

    def abandon(self, operation_id: int, reason: str) -> None:
        """Discard consumption interest while retaining all physical/result obligations.

        Args:
            operation_id: Accepted operation handle, never a request identifier.
            reason: Diagnostic reason supplied by orchestration.
        """
        self._owner()
        record = self.operations.get(operation_id)
        if record is not None:
            self._abandon(record, reason)

    def _abandon(self, record: OperationRecord, reason: str) -> None:
        """Commit abandonment and request supported advisory cancellation once."""
        if not record.abandoned:
            self._count("abandoned")
        record.abandoned = True
        if record.fault is None:
            record.fault = reason
        if record.lifecycle is not None:
            record.lifecycle.request_drain(reason)
        self._mark(record, consume=False)
        binding = self.bindings[record.intent.binding_id]
        if (
            record.attempt is not None
            and not record.cancel_sent
            and not binding.shutdown_started
            and binding.backend is not None
            and binding.backend.capabilities.cancel
        ):
            record.cancel_sent = True
            try:
                record.attempt.cancel()
            except Exception as error:
                self._stop_binding(record.intent.binding_id, f"backend cancel failed: {error}")

    def advance(self, now: float) -> tuple[OperationObservation, ...]:
        """Observe operation facts and perform bounded manager-thread cleanup.

        Args:
            now: Monotonic observation timestamp; safety timing belongs to Lifecycle.

        Returns:
            Changed operation observations, including the final retirement update.
        """
        self._owner()
        for watchdog in {item.lifecycle_binding.watchdog for item in self.bindings.values()}:
            watchdog.progress()
        self._observe_shutdowns()
        updates: list[OperationObservation] = []
        for record in tuple(self.operations.values()):
            self._progress_operation(record)
            retired = self._retire(record)
            state = (id(record.outcome), record.fault, record.marking, record.access_ended, retired)
            if state != record.observed:
                record.observed = state
                updates.append(
                    OperationObservation(
                        record.id,
                        record.outcome,
                        record.fault,
                        record.marking,
                        record.access_ended,
                        retired,
                    )
                )
        return tuple(updates)

    def _progress_operation(self, record: OperationRecord) -> None:
        """Observe delivery and access independently without driving provider I/O."""
        binding = self.bindings[record.intent.binding_id]
        if record.timeout.is_set():
            self._abandon(record, "Lifecycle exposure timeout")
        if record.submission is Submission.NOT_CALLED:
            if not binding.accepting:
                self._abandon(record, "binding admission closed before exposure")
            if not record.abandoned:
                self._poll_lease(record)
                if record.lease_view is not None and not record.abandoned:
                    self._submit_backend(record)
        elif not record.released and not record.unsafe_lease:
            self._poll_lease(record)
        if record.attempt is not None:
            if record.outcome is None:
                try:
                    outcome = record.attempt.poll()
                except Exception as error:
                    self._abandon(record, f"backend outcome query failed: {error}")
                    self._stop_binding(record.intent.binding_id, record.fault or "invalid Attempt")
                else:
                    if outcome is not None:
                        self._observe_outcome(record, outcome)
            if not record.access_ended:
                try:
                    record.access_ended = record.attempt.quiesced() is True
                except Exception as error:
                    self._abandon(record, f"backend access query failed: {error}")
                    self._stop_binding(record.intent.binding_id, record.fault or "invalid Attempt")
        elif record.submission is Submission.REJECTED or (
            record.submission is Submission.NOT_CALLED and record.abandoned
        ):
            record.access_ended = True
        if binding.retained_labels == frozenset():
            record.access_ended = True
        if record.abandoned:
            self._mark(record, consume=False)
        if record.access_ended:
            self._release(record)

    def _poll_lease(self, record: OperationRecord) -> None:
        """Read an unreleased lease on its owner thread without querying readiness."""
        try:
            view = record.lease.poll()
            failure = record.lease.failure
        except Exception as error:
            record.unsafe_lease = True
            self._abandon(record, f"lease poll failed: {error}")
            self._stop_binding(record.intent.binding_id, record.fault or "invalid lease")
            return
        if view is not None and record.lease_view is None:
            record.lease_view = view
        if failure is not None:
            self._abandon(record, failure)

    def _submit_backend(self, record: OperationRecord) -> None:
        """Expose one rooted operation to Lifecycle before its single provider call."""
        binding = self.bindings[record.intent.binding_id]
        try:
            project_view(record.lease_view, binding.namespace)
        except (TypeError, ValueError) as error:
            self._abandon(record, f"invalid lender view: {error}")
            return
        policy = binding.lifecycle_binding
        try:
            record.lifecycle = policy.watchdog.create_owner(
                record.id,
                "receive" if record.intent.direction is Direction.FETCH else "send",
                policy.timeout_s,
            )
        except RuntimeError as error:
            self._abandon(record, str(error))
            return
        record.lifecycle.bind_timeout_outcome(record.timeout.set)
        if not record.lifecycle.expose(record):
            self._abandon(record, "Lifecycle exposure admission closed")
            return
        record.exposed_at = self._clock()
        self._count("submitted")
        try:
            record.attempt = (
                binding.backend.fetch(record.lease_view, record.intent.hint)
                if record.intent.direction is Direction.FETCH
                else binding.backend.publish(record.lease_view, record.intent.hint)
            )
        except SubmissionRejected as error:
            record.submission = Submission.REJECTED
            self._abandon(record, str(error))
        except Exception as error:
            record.submission = Submission.UNCERTAIN
            self._abandon(record, f"uncertain backend submission: {error}")
            self._stop_binding(record.intent.binding_id, record.fault or "uncertain submission")
        else:
            record.submission = Submission.RETURNED

    def _observe_outcome(self, record: OperationRecord, outcome: Outcome) -> None:
        """Retain a fixed validated result and arbitrate marking with Lifecycle."""
        try:
            if not isinstance(outcome, Outcome) or record.lease_view is None:
                raise ValueError("Attempt returned a noncanonical Outcome")
            outcome.check_mask_shapes(record.lease_view)
        except ValueError as error:
            self._abandon(record, f"invalid backend outcome: {error}")
            self._stop_binding(record.intent.binding_id, record.fault or "invalid outcome")
            return
        record.outcome = outcome
        self._count("outcomes")
        if outcome.fault is not None:
            logger.warning(
                f"Shared KV outcome fault: binding={record.intent.binding_id} operation={record.id}: {outcome.fault}"
            )
        if record.lifecycle is not None:
            record.lifecycle.complete_pieces()
        # complete_pieces samples the clock and can commit timeout while poll returns.
        if record.timeout.is_set():
            self._abandon(record, "Lifecycle exposure timeout")
        self._mark(record, consume=not record.abandoned)

    def _mark(self, record: OperationRecord, *, consume: bool) -> None:
        """Make one write decision under the same arbiter that commits expiry."""
        if (
            record.intent.direction is Direction.PUBLISH
            or record.mark_attempted
            or record.lease_view is None
            or record.unsafe_lease
        ):
            return
        policy = self.bindings[record.intent.binding_id].lifecycle_binding
        with policy.watchdog.lock:
            if record.lifecycle is not None:
                record.lifecycle.check()
            if policy.watchdog.fatal is not None:
                return
            if record.timeout.is_set():
                record.abandoned = True
                record.fault = record.fault or "Lifecycle exposure timeout"
            consume = consume and not record.abandoned
            if consume and record.outcome is None:
                return
            masks = (
                record.outcome.delivered_masks if consume else record.lease_view.row_masks(False)
            )
            record.mark_attempted = True
            try:
                record.lease.mark_arrived(masks)
            except Exception as error:
                record.unsafe_lease = True
                record.fault = record.fault or f"lease marking failed: {error}"
                self._stop_binding(record.intent.binding_id, record.fault)
                return
            record.marking = Marking.DELIVERED if consume else Marking.ABANDONED

    def _release(self, record: OperationRecord) -> None:
        """Return protection only after ended access wins Lifecycle arbitration."""
        policy = self.bindings[record.intent.binding_id].lifecycle_binding
        if record.release_attempted or record.unsafe_lease or policy.watchdog.fatal is not None:
            return
        if record.lifecycle is not None and not record.lifecycle.settle(record):
            return
        record.release_attempted = True
        try:
            record.lease.release()
        except Exception as error:
            record.unsafe_lease = True
            record.fault = record.fault or f"lease release failed: {error}"
            if record.lifecycle is not None:
                record.lifecycle.retain_unproven(record, record.fault)
            self._stop_binding(record.intent.binding_id, record.fault)
            return
        record.released = True
        self._count("released")

    def _retire(self, record: OperationRecord) -> bool:
        """Keep final result observation even after abandonment and early release."""
        if not record.released or record.unsafe_lease:
            return False
        if (
            record.intent.direction is Direction.FETCH
            and record.lease_view is not None
            and record.marking is Marking.UNDECIDED
        ):
            return False
        if record.submission is Submission.RETURNED and record.outcome is None:
            return False
        if record.lifecycle is not None and not record.lifecycle.close():
            return False
        del self.operations[record.id]
        self._count("retired")
        return True

    def _binding_deadline(self, binding: BackendBinding, reason: str) -> None:
        """Retain uncertain construction and teardown through the bound Lifecycle."""
        policy = binding.lifecycle_binding
        if binding.lifecycle is None and policy.watchdog.fatal is None:
            try:
                binding.lifecycle = policy.watchdog.create_owner(-1, "receive", policy.timeout_s)
            except RuntimeError:
                return
            binding.lifecycle.expose(binding)
        if binding.lifecycle is not None:
            binding.lifecycle.request_drain(reason)

    def _stop_binding(self, binding_id: str, fault: str) -> None:
        """Close provider admission without asserting a scoped confinement proof."""
        binding = self.bindings[binding_id]
        binding.accepting = False
        binding.fault = fault
        self._binding_deadline(binding, fault)
        self._launch_shutdown(binding, self._clock() + binding.lifecycle_binding.timeout_s)

    def begin_shutdown(self, deadline: float) -> None:
        """Abandon operation interest and start bounded off-loop provider cleanup.

        Args:
            deadline: Absolute monotonic shutdown bound; later calls cannot extend it.

        Raises:
            ValueError: Deadline is not finite.
        """
        self._owner()
        if not math.isfinite(deadline):
            raise ValueError("shutdown deadline must be finite")
        self._shutdown_deadline = (
            deadline if self._shutdown_deadline is None else min(self._shutdown_deadline, deadline)
        )
        for record in tuple(self.operations.values()):
            self._abandon(record, "holder shutdown")
        for binding in self.bindings.values():
            binding.accepting = False
            self._binding_deadline(binding, "backend shutdown")
            self._launch_shutdown(binding, self._shutdown_deadline)

    def _launch_shutdown(self, binding: BackendBinding, deadline: float) -> None:
        """Use one retained serial worker per binding for repeatable SDK cleanup."""
        binding.shutdown_deadline = (
            deadline
            if binding.shutdown_deadline is None
            else min(binding.shutdown_deadline, deadline)
        )
        if binding.backend is None or binding.shutdown_started:
            return
        binding.shutdown_started = True
        backend = binding.backend

        def shutdown() -> None:
            """Publish access evidence only; manager-thread callbacks use dispatch."""
            while not binding.shutdown_stop.is_set():
                try:
                    labels = backend.shutdown(max(0.0, binding.shutdown_deadline - self._clock()))
                    if not isinstance(labels, frozenset):
                        raise TypeError("backend shutdown must return a frozenset")
                except Exception as error:
                    labels, fault = None, f"backend shutdown failed: {error}"
                else:
                    fault = None
                with binding.shutdown_lock:
                    # Only the latest whole-provider evidence is needed; faults remain latched.
                    binding.shutdown_results.append((labels, fault))
                    while len(binding.shutdown_results) > 2:
                        binding.shutdown_results.popleft()
                self.notify()
                if fault is not None or binding.lifecycle_binding.watchdog.fatal is not None:
                    return
                binding.shutdown_stop.wait(0.01)

        binding.shutdown_thread = threading.Thread(
            target=shutdown, name="shared-kv-shutdown", daemon=True
        )
        binding.shutdown_thread.start()

    def _observe_shutdowns(self) -> None:
        """Join provider evidence with bounded owner-finalizer and Lifecycle cleanup."""
        for key, binding in self.bindings.items():
            with binding.shutdown_lock:
                results = tuple(binding.shutdown_results)
                binding.shutdown_results.clear()
            for labels, fault in results:
                binding.retained_labels = labels
                if fault is not None:
                    binding.fault = fault
            policy = binding.lifecycle_binding
            if policy.watchdog.fatal is not None:
                continue
            binding.cleanup.drain()
            if binding.cleanup.fault is not None:
                self._stop_binding(key, binding.cleanup.fault)
            if (
                binding.finalized
                or binding.construction_uncertain
                or binding.retained_labels != frozenset()
            ):
                continue
            try:
                cleanup_complete = binding.provider_cleanup_complete() is True
            except Exception as error:
                binding.cleanup.fault = f"provider cleanup query failed: {error}"
                self._stop_binding(key, binding.cleanup.fault)
                continue
            if cleanup_complete:
                # SDK cleanup is finished. Owner callbacks and lender/Lifecycle
                # retention can continue independently without another SDK call.
                binding.shutdown_stop.set()
            if (
                not cleanup_complete
                or binding.cleanup.fault is not None
                or not binding.cleanup.finish()
            ):
                continue
            # Lender calls later in this round can still fail. Keep the binding's
            # safety owner until every associated operation has completed its calls.
            if any(record.intent.binding_id == key for record in self.operations.values()):
                continue
            if binding.lifecycle is not None and (
                not binding.lifecycle.settle(binding) or not binding.lifecycle.close()
            ):
                continue
            binding.finalized = True
            binding.shutdown_stop.set()

    def pending(self) -> bool:
        """Report retained operations and accepted cleanup requiring owner-loop rounds."""
        return bool(self.operations) or any(
            item.cleanup.pending() or (item.lifecycle is not None and not item.finalized)
            for item in self.bindings.values()
        )

    def shutdown_status(self) -> ShutdownReport:
        """Return completion only after all independent ownership obligations end."""
        retained = tuple(
            key
            for key, binding in self.bindings.items()
            if not binding.finalized
            or any(record.intent.binding_id == key for record in self.operations.values())
        )
        return ShutdownReport(
            self._shutdown_deadline is not None and not retained and not self.operations, retained
        )
