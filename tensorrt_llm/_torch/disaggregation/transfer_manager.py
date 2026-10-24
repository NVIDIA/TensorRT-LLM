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
from typing import TYPE_CHECKING

from tensorrt_llm.logger import logger

from ._assembly import (
    BackendFactory,
    FactoryRejected,
    ProviderCleanupCompleteFn,
    RegionNamespace,
    WriteReturnFn,
    check_view_compatibility,
)
from .base.backend import (
    Attempt,
    Backend,
    Capabilities,
    Hint,
    MemoryKind,
    Outcome,
    SubmissionRejected,
    WakeFn,
)
from .lifecycle.retirement import RetirementDeadline, RetirementWatchdog

if TYPE_CHECKING:
    from ..pyexecutor.kv_cache.sharing import Lease, LeaseView, ResumeBounds


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
    memory_kind: MemoryKind
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


class ConsumptionDecision(Enum):
    """Consumption intent, separate from the result of returning the lease."""

    UNDECIDED = "undecided"
    CONSUME = "consume"
    DISCARD = "discard"


class ReturnStatus(Enum):
    """Whether ownership ended through a consuming return or discard release."""

    PENDING = "pending"
    RETURNED = "returned"
    FAILED = "failed"


@dataclass(frozen=True)
class ReturnDecision:
    """Caller assurance that the complete transfer group permits write return.

    Orchestration settles Outcomes and backend access across all required managers
    and participating ranks before granting this permission. The holder verifies
    local evidence; this flag alone does not establish distributed completion.
    """

    reusable: bool
    all_ranks_returnable: bool

    def __post_init__(self) -> None:
        """Require explicit Boolean consumption and coordination decisions."""
        if type(self.reusable) is not bool or type(self.all_ranks_returnable) is not bool:
            raise TypeError("return decisions require Boolean permissions")


@dataclass(frozen=True)
class OperationObservation:
    """Operation facts only; orchestration owns readiness and request recovery."""

    operation_id: int
    outcome: Outcome | None
    fault: str | None
    decision: ConsumptionDecision
    access_ended: bool
    return_status: ReturnStatus
    resume_bounds: ResumeBounds | None
    retired: bool


@dataclass(frozen=True)
class DomainFailure:
    """Lender-wide stop notification; orchestration owns cache and request recovery."""

    domain_id: str
    fault: str


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
    domain_id: str
    backend: Backend | None = None
    capabilities: Capabilities | None = None
    accepting: bool = True
    construction_uncertain: bool = False
    retained_labels: frozenset | None = None
    finalized: bool = False
    shutdown_started: bool = False
    shutdown_deadline: float | None = None
    shutdown_thread: threading.Thread | None = None
    shutdown_result: _ShutdownObservation | None = None
    shutdown_results_complete: bool = False
    shutdown_cancels: set[int] = field(default_factory=set)
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
    write_return: WriteReturnFn | None = None
    decision: ConsumptionDecision = ConsumptionDecision.UNDECIDED
    return_decision: ReturnDecision | None = None
    return_status: ReturnStatus = ReturnStatus.PENDING
    resume_bounds: ResumeBounds | None = None
    return_attempted: bool = False
    released: bool = False
    release_attempted: bool = False
    cancel_sent: bool = False
    abandoned: bool = False
    access_ended: bool = False
    release_failed: bool = False
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


@dataclass(frozen=True)
class _AttemptObservation:
    """Cumulative provider facts, copied from the worker to the owner loop."""

    operation_id: int
    outcome: Outcome | None
    access_ended: bool
    fault: str | None = None


@dataclass(frozen=True)
class _ShutdownObservation:
    """One bounded snapshot; no provider call holds the mailbox lock."""

    labels: frozenset | None
    attempts: tuple[_AttemptObservation, ...]
    fault: str | None
    results_complete: bool


@dataclass
class _ShutdownAttempt:
    """Provider-call state owned exclusively by the shutdown worker after handoff."""

    operation_id: int
    attempt: Attempt
    outcome: Outcome | None
    access_ended: bool
    cancel_sent: bool
    fault: str | None = None


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
        self._domain_faults: dict[str, str] = {}
        self._domain_updates: deque[DomainFailure] = deque()
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
        domain_id: str,
        lifecycle: LifecycleBinding,
        provider_cleanup_complete: ProviderCleanupCompleteFn,
        config: object = None,
    ) -> None:
        """Root provider construction ownership before invoking its private factory.

        Args:
            binding_id: Unique fixed provider binding identifier.
            namespace: Verified complete regions and assembly-known memory kind.
            domain_id: Opaque lender identity shared by every binding using that lender.
                It identifies failure scope, not a lender service or a memory region.
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
        if not binding_id or binding_id in self.bindings or not profile_id or not domain_id:
            raise ValueError("binding and profile identifiers must be nonempty and unique")
        if domain_id in self._domain_faults:
            raise RuntimeError("backend assembly is unavailable for a failed lender domain")
        cleanup = _OwnerCleanup(self.notify)
        binding = BackendBinding(
            namespace, profile_id, lifecycle, factory, cleanup, provider_cleanup_complete, domain_id
        )
        # Reserve the teardown owner before construction or watchdog admission can
        # close. No claim or active clock is needed while the binding is healthy.
        binding.lifecycle = lifecycle.watchdog.create_owner(-1, "receive", lifecycle.timeout_s)
        self.bindings[binding_id] = binding
        try:
            binding.backend = factory(namespace.regions, self.notify, config, cleanup.post)
            if binding.backend is None:
                raise TypeError("backend factory returned no Backend")
            binding.capabilities = binding.backend.capabilities
            if not isinstance(binding.capabilities, Capabilities):
                raise TypeError("backend returned noncanonical capabilities")
        except FactoryRejected:
            # Certified construction rejection cannot leave accepted cleanup behind.
            if cleanup.pending():
                binding.construction_uncertain = True
                self._stop_binding(binding_id, "rejected constructor left owner cleanup")
            else:
                cleanup.finish()
                binding.lifecycle.close()
                del self.bindings[binding_id]
            raise
        except Exception as error:
            binding.construction_uncertain = True
            self._stop_binding(binding_id, f"backend construction failed: {error}")
            raise

    def submit(
        self, lease: Lease, intent: TransferIntent, *, write_return: WriteReturnFn | None = None
    ) -> SubmitResult:
        """Accept ownership before polling or exposing an acquired lease.

        Args:
            lease: Caller-owned read or write lease, possibly still pending.
            intent: Frozen fixed-binding intent consistent with its lending call.
            write_return: Endpoint bound to the originating lender for FETCH;
                PUBLISH does not use one. The caller proves lease provenance.

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
        if intent.direction is Direction.FETCH and not callable(write_return):
            return Rejected("fetch requires its originating lender's write-return endpoint")
        if intent.direction is Direction.PUBLISH and write_return is not None:
            return Rejected("publish cannot have a write-return endpoint")
        if any(record.lease is lease for record in self.operations.values()):
            return Rejected("lease ownership already transferred")
        watchdog = binding.lifecycle_binding.watchdog
        if (
            self._shutdown_deadline is not None
            or not binding.accepting
            or binding.domain_id in self._domain_faults
            or watchdog.shutdown_requested
            or watchdog.fatal is not None
        ):
            return Deferred("binding admission is closed")
        caps = binding.capabilities
        if not (caps.fetch if intent.direction is Direction.FETCH else caps.publish):
            return Rejected("backend does not support the requested direction")
        self._operation_id += 1
        record = OperationRecord(
            self._operation_id, intent, lease, self._clock(), write_return=write_return
        )
        self.operations[record.id] = record
        self._count("accepted")
        return Accepted(record.id)

    def take_domain_failures(self) -> tuple[DomainFailure, ...]:
        """Drain lender stop notifications without making request recovery decisions."""
        self._owner()
        updates = tuple(self._domain_updates)
        self._domain_updates.clear()
        return updates

    def stop_lender_domain(self, domain_id: str, fault: str) -> None:
        """Stop lender calls after an unexpected lending, polling or readiness error.

        Orchestration uses this for errors in its own lender calls. It must stop
        lending/querying bounds and treat affected fetch intervals as empty.
        The holder still observes provider evidence and releases leases/holds.

        Args:
            domain_id: Assembly identity of the affected lender.
            fault: Diagnostic reason, without any request recovery instructions.
        """
        self._owner()
        if domain_id in self._domain_faults:
            return
        self._domain_faults[domain_id] = fault
        self._domain_updates.append(DomainFailure(domain_id, fault))
        # Publish the stop before abandonment can fail and start provider cleanup.
        for key, binding in self.bindings.items():
            if binding.domain_id == domain_id:
                self._stop_binding(key, fault)
        for record in tuple(self.operations.values()):
            if self.bindings[record.intent.binding_id].domain_id == domain_id:
                self._abandon(record, fault)
        self.notify()

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
        if not record.return_attempted:
            record.decision = ConsumptionDecision.DISCARD
        binding = self.bindings[record.intent.binding_id]
        if (
            record.attempt is not None
            and not record.cancel_sent
            and binding.capabilities is not None
            and binding.capabilities.cancel
        ):
            record.cancel_sent = True
            if binding.shutdown_started:
                with binding.shutdown_lock:
                    binding.shutdown_cancels.add(record.id)
                return
            try:
                record.attempt.cancel()
            except Exception as error:
                self._stop_binding(record.intent.binding_id, f"backend cancel failed: {error}")

    def authorize_return(self, operation_id: int, decision: ReturnDecision) -> None:
        """Return one consuming write after orchestration's collective return gate.

        Args:
            operation_id: Current operation from the coordinated transfer group.
            decision: Reuse permission and assurance that every manager/rank has a
                settled result and ended access. Orchestration prevalidates all
                group inputs and invokes their returns in the same owner iteration.

        Raises:
            KeyError: The operation is unknown or already retired.
            TypeError: The decision has the wrong type.
            ValueError: The decision conflicts or required local/collective evidence
                is absent. These checks perform no lender call.
        """
        self._owner()
        record = self.operations[operation_id]
        if not isinstance(decision, ReturnDecision):
            raise TypeError("write return requires a ReturnDecision")
        if not decision.all_ranks_returnable:
            raise ValueError("write return requires all-rank return assurance")
        if record.intent.direction is not Direction.FETCH:
            raise ValueError("only fetch operations consume write returns")
        if record.return_decision is not None:
            if record.return_decision != decision:
                raise ValueError("conflicting write-return authorization")
            return
        # Consumption cannot resume after discard, even if evidence arrives later.
        if record.abandoned:
            return
        binding = self.bindings[record.intent.binding_id]
        if (
            record.outcome is None
            or not record.access_ended
            or record.write_return is None
            or binding.domain_id in self._domain_faults
        ):
            raise ValueError("write return requires a settled Outcome and ended access")
        policy = binding.lifecycle_binding
        with policy.watchdog.lock:
            if record.lifecycle is not None:
                record.lifecycle.check()
            if record.timeout.is_set():
                self._abandon(record, "Lifecycle exposure timeout")
            if record.abandoned or policy.watchdog.fatal is not None:
                return
            if record.lifecycle is not None and not record.lifecycle.settle(record):
                return
            record.return_decision = decision
            record.decision = ConsumptionDecision.CONSUME
            record.return_attempted = True
            try:
                record.resume_bounds = record.write_return(
                    record.lease, record.outcome.delivered_masks, decision.reusable
                )
            except Exception as error:
                # Return can partly consume a lease. Retain it and never retry or
                # substitute release after the operational call has failed.
                record.return_status = ReturnStatus.FAILED
                record.fault = record.fault or f"write return failed: {error}"
                if record.lifecycle is not None:
                    record.lifecycle.retain_unproven(record, record.fault)
                self.stop_lender_domain(binding.domain_id, record.fault)
                return
            record.released = True
            record.return_status = ReturnStatus.RETURNED
            self._count("returned")

    def advance(self, now: float) -> tuple[OperationObservation, ...]:
        """Observe operation facts and perform bounded manager-thread cleanup.

        Orchestration calls each active lender's progress once per executor
        iteration before this method. Keep that progress running through idle and
        shutdown rounds, including pending copies after a consuming lease return.
        This holder polls ready facts and never advances lender or backend I/O.

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
            state = (
                id(record.outcome),
                record.fault,
                record.decision,
                record.access_ended,
                record.return_status,
                record.resume_bounds,
                retired,
            )
            if state != record.observed:
                record.observed = state
                updates.append(
                    OperationObservation(
                        record.id,
                        record.outcome,
                        record.fault,
                        record.decision,
                        record.access_ended,
                        record.return_status,
                        record.resume_bounds,
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
        if record.attempt is not None and not binding.shutdown_started:
            if record.outcome is None:
                try:
                    outcome = record.attempt.poll()
                except Exception as error:
                    self._abandon(record, f"backend outcome query failed: {error}")
                    self._stop_binding(record.intent.binding_id, record.fault or "invalid Attempt")
                else:
                    if outcome is not None:
                        self._observe_outcome(record, outcome)
            # A failed poll can hand off this binding while
            # processing the outcome. Do not resume provider calls afterward.
            if not record.access_ended and not binding.shutdown_started:
                try:
                    record.access_ended = record.attempt.quiesced() is True
                except Exception as error:
                    self._abandon(record, f"backend access query failed: {error}")
                    self._stop_binding(record.intent.binding_id, record.fault or "invalid Attempt")
        if record.submission is Submission.REJECTED or (
            record.submission is Submission.NOT_CALLED and record.abandoned
        ):
            record.access_ended = True
        if binding.retained_labels == frozenset():
            record.access_ended = True
        if record.access_ended and (
            record.intent.direction is Direction.PUBLISH or record.abandoned
        ):
            self._release(record)

    def _poll_lease(self, record: OperationRecord) -> None:
        """Poll only until a view is ready; lender progress belongs to orchestration."""
        if record.lease_view is not None or record.released:
            return
        binding = self.bindings[record.intent.binding_id]
        if binding.domain_id in self._domain_faults:
            return
        try:
            view = record.lease.poll()
            failure = record.lease.failure
        except Exception as error:
            self.stop_lender_domain(binding.domain_id, f"lease poll failed: {error}")
            return
        if view is not None and record.lease_view is None:
            record.lease_view = view
        if failure is not None:
            self._abandon(record, failure)

    def _submit_backend(self, record: OperationRecord) -> None:
        """Expose one rooted operation to Lifecycle before its single provider call."""
        binding = self.bindings[record.intent.binding_id]
        try:
            check_view_compatibility(record.lease_view, binding.namespace)
        except ValueError as error:
            self._abandon(record, f"incompatible lender view: {error}")
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
        """Retain the fixed result without granting consumption or release."""
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

    def _release(self, record: OperationRecord) -> None:
        """Return protection only after ended access wins Lifecycle arbitration."""
        policy = self.bindings[record.intent.binding_id].lifecycle_binding
        if (
            record.return_attempted
            or record.release_attempted
            or record.released
            or policy.watchdog.fatal is not None
        ):
            return
        if record.lifecycle is not None and not record.lifecycle.settle(record):
            return
        record.release_attempted = True
        try:
            record.lease.release()
        except Exception as error:
            record.release_failed = True
            record.return_status = ReturnStatus.FAILED
            record.fault = record.fault or f"lease release failed: {error}"
            if record.lifecycle is not None:
                record.lifecycle.retain_unproven(record, record.fault)
            self.stop_lender_domain(self.bindings[record.intent.binding_id].domain_id, record.fault)
            return
        record.released = True
        record.return_status = ReturnStatus.RETURNED
        self._count("released")

    def _retire(self, record: OperationRecord) -> bool:
        """Keep final result observation even after abandonment and early release."""
        if not record.released or record.return_status is ReturnStatus.FAILED:
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
        # This existing owner survives shutdown admission closure. Retaining it
        # starts grace, preserving any earlier watchdog-wide drain deadline.
        binding.lifecycle.retain_unproven(binding, reason)

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
        """Hand every provider call to one worker without blocking the owner loop."""
        binding.shutdown_deadline = (
            deadline
            if binding.shutdown_deadline is None
            else min(binding.shutdown_deadline, deadline)
        )
        if binding.backend is None or binding.shutdown_started:
            return
        binding.accepting = False
        binding.shutdown_started = True
        backend = binding.backend
        attempts = tuple(
            _ShutdownAttempt(
                record.id,
                record.attempt,
                record.outcome,
                record.access_ended,
                record.cancel_sent,
            )
            for record in self.operations.values()
            if self.bindings[record.intent.binding_id] is binding and record.attempt is not None
        )

        def shutdown() -> None:
            """Serialize SDK calls and publish cumulative facts; never call a lease."""
            labels: frozenset | None = None
            fault: str | None = None
            while not binding.shutdown_stop.is_set():
                with binding.shutdown_lock:
                    cancels = binding.shutdown_cancels
                    binding.shutdown_cancels = set()
                for state in attempts:
                    if state.operation_id in cancels and not state.cancel_sent:
                        state.cancel_sent = True
                        try:
                            state.attempt.cancel()
                        except Exception as error:
                            state.fault = state.fault or f"backend cancel failed: {error}"
                try:
                    observed = backend.shutdown(max(0.0, binding.shutdown_deadline - self._clock()))
                    if not isinstance(observed, frozenset):
                        raise TypeError("backend shutdown must return a frozenset")
                except Exception as error:
                    fault = fault or f"backend shutdown failed: {error}"
                else:
                    if labels != frozenset():
                        labels = observed
                # Shutdown forces final Outcomes. Read them before publishing its
                # access evidence, including for leases already safely returned.
                for state in attempts:
                    if state.outcome is None:
                        try:
                            state.outcome = state.attempt.poll()
                        except Exception as error:
                            state.fault = state.fault or f"backend outcome query failed: {error}"
                    if not state.access_ended:
                        try:
                            state.access_ended = state.attempt.quiesced() is True
                        except Exception as error:
                            state.fault = state.fault or f"backend access query failed: {error}"
                snapshot = _ShutdownObservation(
                    labels,
                    tuple(
                        _AttemptObservation(
                            state.operation_id, state.outcome, state.access_ended, state.fault
                        )
                        for state in attempts
                    ),
                    fault,
                    all(state.outcome is not None for state in attempts),
                )
                with binding.shutdown_lock:
                    # One cumulative snapshot bounds memory even if the owner is idle.
                    binding.shutdown_result = snapshot
                self.notify()
                if binding.lifecycle_binding.watchdog.fatal is not None:
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
                result = binding.shutdown_result
                binding.shutdown_result = None
            if result is not None:
                binding.retained_labels = result.labels
                binding.shutdown_results_complete = result.results_complete
                if result.fault is not None:
                    binding.fault = result.fault
                for observed in result.attempts:
                    record = self.operations.get(observed.operation_id)
                    if record is None:
                        continue
                    if observed.fault is not None:
                        self._abandon(record, observed.fault)
                    if observed.outcome is not None and record.outcome is None:
                        self._observe_outcome(record, observed.outcome)
                    record.access_ended = record.access_ended or observed.access_ended
            policy = binding.lifecycle_binding
            if policy.watchdog.fatal is not None:
                continue
            binding.cleanup.drain()
            if binding.cleanup.fault is not None:
                # Manager-thread finalizers can release lender-owned holds. An
                # unexpected failure closes the same lender domain as lease errors.
                self.stop_lender_domain(binding.domain_id, binding.cleanup.fault)
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
            if cleanup_complete and binding.shutdown_results_complete:
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
            item.cleanup.pending() or (not item.accepting and not item.finalized)
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
