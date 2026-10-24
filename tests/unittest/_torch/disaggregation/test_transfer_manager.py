# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Holder contracts; doubles isolate only external I/O and lender-owned CUDA copies."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation._assembly import FactoryRejected, RegionNamespace
from tensorrt_llm._torch.disaggregation.base.backend import (
    Capabilities,
    Outcome,
    SubmissionRejected,
)
from tensorrt_llm._torch.disaggregation.lifecycle.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.transfer_manager import (
    Accepted,
    ConsumptionDecision,
    Deferred,
    Direction,
    LifecycleBinding,
    Rejected,
    ReturnDecision,
    ReturnStatus,
    Submission,
    TransferIntent,
    TransferManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LeaseView,
    LentLayerGroupRows,
    MemoryKind,
    MemoryRegion,
    ResumeBounds,
)

pytestmark = pytest.mark.cpu_only


@dataclass
class _Lease:
    """Lender boundary with explicit copy readiness and one-shot call assertions."""

    view: LeaseView
    ready: bool = True
    failure: str | None = None
    released: bool = False
    polls: int = 0
    returns: list[tuple[tuple[np.ndarray, ...], bool]] = field(default_factory=list)
    bounds: ResumeBounds | None = ResumeBounds(4, 0)
    actions: list[str] = field(default_factory=list)
    poll_error: Exception | None = None
    return_error: Exception | None = None
    release_error: Exception | None = None

    def poll(self) -> LeaseView | None:
        """Return the ready protected view, detecting invalid polling after release."""
        assert not self.released
        self.polls += 1
        if self.poll_error is not None:
            raise self.poll_error
        return self.view if self.ready and self.failure is None else None

    def release(self) -> None:
        """Record protection return without claiming local copies have completed."""
        assert not self.released
        self.actions.append("release")
        if self.release_error is not None:
            raise self.release_error
        self.released = True


def _return_write(
    lease: _Lease, masks: tuple[np.ndarray, ...], reusable: bool
) -> ResumeBounds | None:
    """Model one consuming call at the lender endpoint, separate from discard release."""
    assert not lease.released and not lease.returns
    lease.returns.append((tuple(mask.copy() for mask in masks), reusable))
    lease.actions.append("return")
    if lease.return_error is not None:
        raise lease.return_error
    lease.released = True
    return lease.bounds


@dataclass
class _Attempt:
    """Independent result and access-end observations, without physical progress."""

    outcome: Outcome | None = None
    ended: bool = False
    cancels: int = 0
    before_poll: Callable[[], None] | None = None
    backend: _Backend | None = None

    def poll(self) -> Outcome | None:
        """Return the fixed result, optionally crossing a deterministic deadline."""
        with self.backend.call("poll"):
            if self.before_poll is not None:
                self.before_poll()
            return self.outcome

    def quiesced(self) -> bool:
        """Return independent whole-Attempt access evidence."""
        with self.backend.call("quiesced"):
            return self.ended

    def cancel(self) -> None:
        """Count advisory cancellation without inventing completion."""
        with self.backend.call("cancel"):
            self.cancels += 1


class _Backend:
    """Controlled provider with observable shutdown threading and owner finalization."""

    def __init__(self, *, cancel: bool = True) -> None:
        """Initialize independently controlled operations and retained registration."""
        self.capabilities = Capabilities(True, True, cancel)
        self.attempts: list[_Attempt] = []
        self.submissions: list[tuple[LeaseView, object, str]] = []
        self.error: Exception | None = None
        self.labels: frozenset = frozenset({("registration", 1)})
        self.shutdown_threads: list[int] = []
        self.shutdown_gate: threading.Event | None = None
        self.shutdown_started = threading.Event()
        self.post = None
        self.posted = False
        self.finalizations: list[int] = []
        self.call_lock = threading.Lock()
        self.calls: list[tuple[str, int]] = []
        self.call_violations: list[str] = []
        self.shutdown_owner: int | None = None

    @contextmanager
    def call(self, name: str) -> Iterator[None]:
        """Enforce one provider call sequence, including thread handoff at shutdown."""
        identity = threading.get_ident()
        if not self.call_lock.acquire(blocking=False):
            self.call_violations.append(f"overlapping {name}")
            raise AssertionError("overlapping provider calls")
        try:
            if self.shutdown_owner is not None and identity != self.shutdown_owner:
                self.call_violations.append(f"owner called {name} after handoff")
                raise AssertionError("provider call after handoff")
            if name == "shutdown":
                self.shutdown_owner = identity
            self.calls.append((name, identity))
            yield
        finally:
            self.call_lock.release()

    def _submit(self, view: LeaseView, hint: object, direction: str) -> _Attempt:
        """Return a fixed handle or inject an explicit submission disposition."""
        self.submissions.append((view, hint, direction))
        if self.error is not None:
            raise self.error
        attempt = _Attempt(backend=self)
        self.attempts.append(attempt)
        return attempt

    def fetch(self, view: LeaseView, source: object) -> _Attempt:
        """Accept destination writes without completing them."""
        with self.call("fetch"):
            return self._submit(view, source, "fetch")

    def publish(self, view: LeaseView, destination: object) -> _Attempt:
        """Accept source reads without requiring an inference request."""
        with self.call("publish"):
            return self._submit(view, destination, "publish")

    def shutdown(self, timeout: float) -> frozenset:
        """Force fixed Outcomes but keep access evidence independently controlled."""
        with self.call("shutdown"):
            self.shutdown_threads.append(threading.get_ident())
            self.shutdown_started.set()
            if self.shutdown_gate is not None:
                self.shutdown_gate.wait(timeout)
            for attempt, (view, _, _) in zip(self.attempts, self.submissions):
                if attempt.outcome is None:
                    attempt.outcome = Outcome.for_view(view, fault="shutdown")
            if not self.labels and not self.posted:
                self.posted = self.post(lambda: self.finalizations.append(threading.get_ident()))
            return self.labels


class _Harness:
    """Real holder and Lifecycle with canonical lender metadata."""

    def __init__(self, *, cancel: bool = True) -> None:
        """Construct the explicitly known HOST fixture without claiming activation."""
        self.now = 10.0
        self.fatals: list[object] = []
        self.watchdog = RetirementWatchdog(self.fatals.append, clock=lambda: self.now)
        self.wake = threading.Event()
        self.manager = TransferManager(wake=self.wake.set, clock=lambda: self.now)
        self.namespace = RegionNamespace(
            (MemoryRegion("layout", 4096, 64, 16, 4, (7,), MemoryKind.HOST),),
            MemoryKind.HOST,
        )
        self.view = LeaseView(
            (
                LentLayerGroupRows(
                    7,
                    np.array([2, 3], dtype=np.int64),
                    np.zeros((2, 54), dtype=np.uint8),
                    np.array([4096, 4128], dtype=np.int64),
                    0,
                ),
            ),
            0,
            4,
        )
        self.backend = _Backend(cancel=cancel)
        self.policy = LifecycleBinding(self.watchdog, 5)
        self.bind("store", self.backend)
        self.intent = TransferIntent("store", "m1", Direction.FETCH, MemoryKind.HOST)

    def bind(self, name: str, backend: _Backend, *, domain_id: str = "lender") -> None:
        """Create an independent fixed binding using the real private factory seam."""

        def factory(regions: object, wake: object, config: object, post: object) -> _Backend:
            """Pass the complete namespace and retain owner-thread dispatch."""
            assert regions is self.namespace.regions
            backend.post = post
            return backend

        self.manager.assemble(
            name,
            self.namespace,
            "m1",
            factory,
            domain_id=domain_id,
            lifecycle=self.policy,
            provider_cleanup_complete=lambda: backend.posted,
        )

    def submit(
        self, *, direction: Direction = Direction.FETCH, ready: bool = True
    ) -> tuple[_Lease, int]:
        """Acquire externally and hand off without any Request/lender dependency."""
        lease = _Lease(self.view, ready=ready)
        result = self.manager.submit(
            lease,
            replace(self.intent, direction=direction),
            write_return=_return_write if direction is Direction.FETCH else None,
        )
        assert isinstance(result, Accepted)
        return lease, result.operation_id

    def tick(self) -> tuple:
        """Run the actual owner-loop holder progress method."""
        return self.manager.advance(self.now)

    def complete(self, *, delivered: bool = True, fault: str | None = None) -> None:
        """Publish one fixed provider Outcome independently from access end."""
        self.backend.attempts[0].outcome = Outcome.for_view(
            self.view, self.view.row_masks(delivered), fault
        )

    def __enter__(self) -> _Harness:
        """Keep test-owned worker cleanup within pytest's measured call scope."""
        return self

    def __exit__(self, *args: object) -> None:
        """Finish or contain test operations before the thread-leak check runs."""
        self.close()

    def close(self) -> None:
        """End fake I/O, finish real cleanup, or contain intentionally violated contracts."""
        for binding in self.manager.bindings.values():
            if isinstance(binding.backend, _Backend):
                binding.backend.labels = frozenset()
                if binding.backend.shutdown_gate is not None:
                    binding.backend.shutdown_gate.set()
        self.manager.begin_shutdown(self.now + 5)
        stop = time.monotonic() + 0.2
        while self.manager.pending() and not self.watchdog.fatal and time.monotonic() < stop:
            self.tick()
            self.wake.wait(0.001)
            self.wake.clear()
        if self.manager.pending() and not self.watchdog.fatal:
            # Malformed outcomes and failed lender/finalizer calls intentionally
            # cannot be repaired. Exercise real conservative containment instead
            # of erasing retained roots or forcing the worker's stop flag.
            self.now += 2 * self.policy.timeout_s + 1
            self.watchdog.progress()
            assert self.watchdog.fatal is not None
        for binding in self.manager.bindings.values():
            if binding.shutdown_thread is not None:
                binding.shutdown_thread.join(timeout=1)
                assert not binding.shutdown_thread.is_alive()
            if isinstance(binding.backend, _Backend):
                assert not binding.backend.call_violations


@pytest.fixture
def h() -> _Harness:
    """Create a harness whose test-body context owns all asynchronous cleanup."""
    return _Harness()


def test_acceptance_roots_pending_lease_without_poll_or_exposure(h: _Harness) -> None:
    """Accepted transfers ownership before the first side effect."""
    with h:
        lease, operation = h.submit(ready=False)
        assert h.manager.operations[operation].lease is lease
        assert lease.polls == 0 and not h.backend.submissions
        h.tick()
        assert lease.polls == 1 and not h.backend.submissions
        h.now = 100
        h.tick()
        assert not h.fatals
        lease.ready = True
        h.tick()
        assert h.backend.submissions[0][0] is h.view
        assert h.manager.operations[operation].exposed_at == 100


@pytest.mark.parametrize(
    "field,value",
    [
        ("binding_id", "missing"),
        ("profile_id", "other"),
        ("memory_kind", MemoryKind.DEVICE),
        ("memory_kind", "HOST"),
        ("direction", "invalid"),
    ],
)
def test_rejected_intent_leaves_lease_untouched(h: _Harness, field: str, value: object) -> None:
    """Metadata mismatch never acquires lease ownership or substitutes a source."""
    with h:
        lease = _Lease(h.view)
        result = h.manager.submit(
            lease, replace(h.intent, **{field: value}), write_return=_return_write
        )
        assert isinstance(result, Rejected)
        assert not h.manager.operations and lease.polls == 0 and not lease.released


def test_deferred_binding_leaves_caller_ownership(h: _Harness) -> None:
    """Admission refusal must not poll, consume or return the caller's lease."""
    with h:
        h.manager.bindings["store"].accepting = False
        lease = _Lease(h.view)
        assert isinstance(h.manager.submit(lease, h.intent, write_return=_return_write), Deferred)
        assert lease.polls == 0 and not lease.actions and not h.manager.operations


def test_same_lease_cannot_have_two_accepted_operations(h: _Harness) -> None:
    """M1 protects one Lease/Attempt owner and rejects accidental active reuse."""
    with h:
        lease, _ = h.submit()
        assert isinstance(h.manager.submit(lease, h.intent, write_return=_return_write), Rejected)
        assert len(h.manager.operations) == 1


@pytest.mark.parametrize("outcome_first", [False, True])
def test_delivery_and_access_end_are_independent(h: _Harness, outcome_first: bool) -> None:
    """Neither evidence order consumes or releases a fetch before coordinated return."""
    with h:
        lease, operation = h.submit()
        h.tick()
        attempt = h.backend.attempts[0]
        if outcome_first:
            h.complete()
            (update,) = h.tick()
            assert update.outcome is attempt.outcome and not update.access_ended
            attempt.ended = True
        else:
            attempt.ended = True
            (update,) = h.tick()
            assert update.access_ended and update.outcome is None
            h.complete()
        assert not lease.released and not lease.returns
        assert operation in h.manager.operations
        h.tick()
        assert not lease.released and not lease.returns
        h.manager.authorize_return(
            operation, ReturnDecision(reusable=True, all_ranks_returnable=True)
        )
        (update,) = h.tick()
        assert update.retired and update.outcome is attempt.outcome
        assert update.decision is ConsumptionDecision.CONSUME
        assert update.return_status is ReturnStatus.RETURNED
        assert update.resume_bounds == lease.bounds
        assert lease.released and lease.returns[0][0][0].all() and lease.returns[0][1]
        assert lease.actions == ["return"]
        assert lease.polls == 1
        assert h.tick() == () and not h.manager.operations


@pytest.mark.parametrize("direction", list(Direction))
def test_direction_requires_matching_return_endpoint(h: _Harness, direction: Direction) -> None:
    """Invalid handoff provenance leaves the lease and its pending work with the caller."""
    with h:
        lease = _Lease(h.view)
        result = h.manager.submit(
            lease,
            replace(h.intent, direction=direction),
            write_return=None if direction is Direction.FETCH else _return_write,
        )
        assert isinstance(result, Rejected)
        assert not h.manager.operations and not lease.actions and lease.polls == 0


@pytest.mark.parametrize("evidence", ["neither", "outcome", "access"])
def test_return_rejects_incomplete_local_evidence(h: _Harness, evidence: str) -> None:
    """Caller assurance cannot replace the holder's fixed result and ended-access checks."""
    with h:
        lease, operation = h.submit()
        h.tick()
        if evidence == "outcome":
            h.complete()
        elif evidence == "access":
            h.backend.attempts[0].ended = True
        h.tick()
        with pytest.raises(ValueError):
            h.manager.authorize_return(operation, ReturnDecision(True, True))
        assert not lease.released and not lease.returns


@pytest.mark.parametrize("reusable", [False, True])
def test_return_requires_collective_assurance_and_is_one_shot(h: _Harness, reusable: bool) -> None:
    """Fixed local evidence permits one authorized call with the caller's reuse permission."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.complete()
        h.backend.attempts[0].ended = True
        h.tick()
        with pytest.raises(ValueError):
            h.manager.authorize_return(operation, ReturnDecision(reusable, False))
        assert not lease.returns
        decision = ReturnDecision(reusable, True)
        h.manager.authorize_return(operation, decision)
        h.manager.authorize_return(operation, decision)
        with pytest.raises(ValueError):
            h.manager.authorize_return(operation, ReturnDecision(not reusable, True))
        assert len(lease.returns) == 1 and lease.returns[0][1] is reusable
        assert lease.actions == ["return"]
        h.tick()
        with pytest.raises(KeyError):
            h.manager.authorize_return(operation, decision)


def test_settled_miss_uses_authorized_return_instead_of_discard(h: _Harness) -> None:
    """A final miss still returns canonical all-false masks to settle the lender fetch."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.complete(delivered=False)
        h.backend.attempts[0].ended = True
        h.tick()
        assert not lease.released and not lease.returns
        h.manager.authorize_return(operation, ReturnDecision(False, True))
        (update,) = h.tick()
        assert update.return_status is ReturnStatus.RETURNED and update.retired
        assert len(lease.returns) == 1 and not lease.returns[0][0][0].any()
        assert not lease.returns[0][1] and lease.actions == ["return"]


def test_abandonment_prevents_late_authorization_from_consuming(h: _Harness) -> None:
    """A completed result cannot restore consumption after orchestration abandons it."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.complete()
        h.backend.attempts[0].ended = True
        h.tick()
        h.manager.abandon(operation, "request cancelled")
        h.manager.authorize_return(operation, ReturnDecision(True, True))
        (update,) = h.tick()
        assert update.retired and update.decision is ConsumptionDecision.DISCARD
        assert lease.released and not lease.returns and lease.actions == ["release"]


def test_abandonment_after_return_keeps_consumption_record_and_avoids_second_return(
    h: _Harness,
) -> None:
    """Late loss of request interest cannot undo consumed rows or return ownership twice."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.complete()
        h.backend.attempts[0].ended = True
        h.tick()
        decision = ReturnDecision(True, True)
        h.manager.authorize_return(operation, decision)
        h.manager.abandon(operation, "request exited after return")
        h.manager.authorize_return(operation, decision)
        (update,) = h.tick()
        assert update.retired and update.decision is ConsumptionDecision.CONSUME
        assert update.return_status is ReturnStatus.RETURNED
        assert len(lease.returns) == 1 and lease.actions == ["return"]


def test_ready_lease_is_never_polled_again(h: _Harness) -> None:
    """Once the view exists, provider progress uses the retained view and Attempt only."""
    with h:
        lease, operation = h.submit()
        h.tick()
        lease.poll_error = AssertionError("ready lease must not be polled twice")
        h.tick()
        h.complete()
        h.backend.attempts[0].ended = True
        h.tick()
        h.manager.authorize_return(operation, ReturnDecision(True, True))
        h.tick()
        assert lease.polls == 1 and lease.released
        assert h.manager.take_domain_failures() == ()


def test_timeout_between_observation_and_authorization_prevents_return(h: _Harness) -> None:
    """Return arbitration rechecks the safety deadline after orchestration observes results."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.complete()
        h.backend.attempts[0].ended = True
        h.tick()
        h.now = 15.0
        h.manager.authorize_return(operation, ReturnDecision(True, True))
        h.tick()
        assert not lease.returns


@pytest.mark.parametrize(
    "direction,abandoned",
    [(Direction.PUBLISH, False), (Direction.PUBLISH, True), (Direction.FETCH, True)],
)
def test_late_result_survives_early_release_even_after_abandonment(
    h: _Harness, direction: Direction, abandoned: bool
) -> None:
    """Publication and abandoned fetch retain their eventual result and provider fault."""
    with h:
        lease, operation = h.submit(direction=direction)
        h.tick()
        if abandoned:
            h.manager.abandon(operation, "caller no longer needs acquisition")
        h.backend.attempts[0].ended = True
        h.tick()
        assert lease.released and operation in h.manager.operations
        h.complete(fault="late provider fault")
        (update,) = h.tick()
        assert update.retired and update.outcome.fault == "late provider fault"
        assert update.fault != "late provider fault"
        assert not lease.returns and lease.actions == ["release"]


def test_unseen_abandonment_needs_no_consumption_or_submission(h: _Harness) -> None:
    """An unpolled accepted lease can be abandoned without inventing a view."""
    with h:
        lease, operation = h.submit(ready=False)
        h.manager.abandon(operation, "request cancelled")
        (update,) = h.tick()
        assert update.retired and update.decision is ConsumptionDecision.DISCARD
        assert lease.released and lease.polls == 0 and not lease.returns
        assert not h.backend.submissions


@pytest.mark.parametrize("supported", [False, True])
def test_cancellation_is_advisory_and_one_shot(supported: bool) -> None:
    """Cancel neither proves access end nor changes a fixed consumption decision."""
    with _Harness(cancel=supported) as h:
        lease, operation = h.submit()
        h.tick()
        h.manager.abandon(operation, "acquisition abandoned")
        h.manager.abandon(operation, "duplicate")
        h.tick()
        assert h.backend.attempts[0].cancels == int(supported)
        assert not lease.released and not lease.returns


@pytest.mark.parametrize(
    "error,safe",
    [(SubmissionRejected("no access"), True), (RuntimeError("may have escaped"), False)],
)
def test_submission_rejection_differs_from_uncertainty(
    h: _Harness, error: Exception, safe: bool
) -> None:
    """Only explicit rejection grants no-access evidence when no Attempt returned."""
    with h:
        h.backend.error = error
        lease, operation = h.submit()
        (update,) = h.tick()
        assert update.retired is safe and lease.released is safe
        assert not lease.returns
        if not safe:
            assert h.manager.operations[operation].submission is Submission.UNCERTAIN
            assert not h.manager.bindings["store"].accepting


def test_malformed_outcome_never_consumes_rows(h: _Harness) -> None:
    """Masks must match the submitted canonical group order and row counts."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.backend.attempts[0].outcome = Outcome((np.ones(3, dtype=bool),))
        h.tick()
        assert not lease.returns
        assert h.manager.operations[operation].outcome is None


def test_timeout_committed_inside_poll_prevents_consumption(h: _Harness) -> None:
    """A poll crossing the exact exposure deadline cannot consume late rows."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.complete()
        h.backend.attempts[0].before_poll = lambda: setattr(h, "now", 15.0)
        (update,) = h.tick()
        assert update.decision is ConsumptionDecision.DISCARD
        assert not lease.returns
        assert h.manager.operations[operation].abandoned


def test_fatal_expiry_cannot_be_undone_by_late_evidence(h: _Harness) -> None:
    """The implemented conservative Lifecycle adapter retains roots after fatal expiry."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.now = 20
        h.watchdog.progress()
        h.complete()
        h.backend.attempts[0].ended = True
        h.tick()
        assert h.fatals and not lease.released and operation in h.manager.operations
        assert not lease.returns


@pytest.mark.parametrize("method", ["poll", "return"])
def test_lender_failure_stops_calls_but_releases_after_access_ends(
    h: _Harness, method: str
) -> None:
    """A failed poll permits safe discard; a failed consuming return is not retried."""
    with h:
        lease, operation = h.submit()
        setattr(lease, method + "_error", RuntimeError("manager failed"))
        h.tick()
        if method == "return":
            h.complete()
            h.backend.attempts[0].ended = True
            h.tick()
            decision = ReturnDecision(reusable=True, all_ranks_returnable=True)
            h.manager.authorize_return(operation, decision)
            h.manager.authorize_return(operation, decision)
            h.tick()
            assert not lease.released and operation in h.manager.operations
            assert h.manager.operations[operation].return_status is ReturnStatus.FAILED
            assert lease.actions == ["return"]
        else:
            assert lease.released and operation not in h.manager.operations
            assert lease.polls == 1 and lease.actions == ["release"]
        assert not h.manager.bindings["store"].accepting
        (failure,) = h.manager.take_domain_failures()
        assert failure.domain_id == "lender" and "manager failed" in failure.fault
        assert h.manager.take_domain_failures() == ()


def test_release_failure_retains_owner_and_closes_lender_domain(h: _Harness) -> None:
    """A failed release can have partial effects and is not retried or erased."""
    with h:
        lease, operation = h.submit()
        lease.release_error = RuntimeError("manager failed")
        h.tick()
        h.manager.abandon(operation, "discard failed acquisition")
        h.complete()
        h.backend.attempts[0].ended = True
        h.tick()
        assert not lease.released and operation in h.manager.operations
        h.tick()
        assert lease.actions.count("release") == 1
        assert h.manager.operations[operation].release_failed
        assert h.manager.take_domain_failures()[0].domain_id == "lender"


def test_documented_lease_failure_before_view_can_retire(h: _Harness) -> None:
    """A canonical failed lease has no backend exposure and needs no consuming return."""
    with h:
        lease, _ = h.submit()
        lease.failure = "staging capacity unavailable"
        (update,) = h.tick()
        assert update.retired and update.fault == lease.failure
        assert lease.released and not lease.returns


def test_constructor_uncertainty_retains_factory_and_accepted_cleanup(h: _Harness) -> None:
    """A missing Backend handle cannot erase its partial construction owner."""
    with h:
        finalized = []

        def factory(regions: object, wake: object, config: object, post: object) -> None:
            """Retain an owner callback before an uncertain constructor failure."""
            assert post(lambda: finalized.append(threading.get_ident()))
            raise RuntimeError("registration may exist")

        with pytest.raises(RuntimeError, match="registration may exist"):
            h.manager.assemble(
                "failed",
                h.namespace,
                "m1",
                factory,
                domain_id="lender",
                lifecycle=h.policy,
                provider_cleanup_complete=lambda: True,
            )
        h.tick()
        binding = h.manager.bindings["failed"]
        assert binding.factory is factory and binding.construction_uncertain
        assert finalized == [threading.get_ident()]
        assert not binding.finalized and not h.manager.shutdown_status().complete


def test_certified_constructor_rejection_does_not_leave_binding(h: _Harness) -> None:
    """Only FactoryRejected certifies that no private construction root is needed."""
    with h:
        watchdog = RetirementWatchdog(h.fatals.append, clock=lambda: h.now)

        def factory(*args: object) -> None:
            """Reject before any resource or callback escapes."""
            raise FactoryRejected("nothing retained")

        with pytest.raises(FactoryRejected):
            h.manager.assemble(
                "rejected",
                h.namespace,
                "m1",
                factory,
                domain_id="lender",
                lifecycle=LifecycleBinding(watchdog, 5),
                provider_cleanup_complete=lambda: True,
            )
        assert "rejected" not in h.manager.bindings
        assert not watchdog.has_unclosed_owners
        watchdog.stop()


def test_idle_binding_has_no_active_deadline_or_pending_work(h: _Harness) -> None:
    """The reserved teardown owner stays dormant throughout healthy idle time."""
    with h:
        assert h.manager.bindings["store"].lifecycle is not None
        assert h.watchdog.has_unclosed_owners
        assert not h.manager.pending()
        h.now = 100
        h.tick()
        assert not h.manager.pending() and not h.fatals


@pytest.mark.parametrize("watchdog_first", [False, True])
def test_idle_shutdown_closes_reserved_owner_after_confirmed_cleanup(
    h: _Harness, watchdog_first: bool
) -> None:
    """Either shutdown order closes the dormant owner only after provider cleanup."""
    with h:
        h.backend.labels = frozenset()
        if watchdog_first:
            h.watchdog.request_shutdown()
        h.manager.begin_shutdown(h.now + 5)
        if not watchdog_first:
            h.watchdog.request_shutdown()
        stop = time.monotonic() + 1
        while not h.manager.shutdown_status().complete and time.monotonic() < stop:
            h.tick()
            h.wake.wait(0.001)
            h.wake.clear()
        assert h.manager.shutdown_status().complete and h.backend.finalizations
        assert not h.manager.pending() and not h.watchdog.has_unclosed_owners
        h.watchdog.stop()


@pytest.mark.parametrize("watchdog_first", [False, True])
def test_idle_shutdown_remains_pending_and_bounded_after_admission_closes(
    h: _Harness, watchdog_first: bool
) -> None:
    """Either shutdown order retains the binding and preserves the first grace clock."""
    with h:
        h.backend.shutdown_gate = threading.Event()
        if watchdog_first:
            h.watchdog.request_shutdown()
            h.now = 12
            h.manager.begin_shutdown(h.now + 5)
        else:
            h.manager.begin_shutdown(h.now + 5)
            h.now = 12
            h.watchdog.request_shutdown()
        assert h.backend.shutdown_started.wait(1)
        assert h.manager.pending() and not h.manager.shutdown_status().complete
        h.now = 14
        h.tick()
        assert h.manager.pending() and not h.fatals
        h.now = 15
        h.tick()
        assert h.fatals and h.manager.pending()
        assert not h.manager.bindings["store"].finalized


def test_shutdown_worker_is_reused_and_owner_finalizers_are_dispatched(h: _Harness) -> None:
    """Repeated retained-label results use one worker and owner-only hold cleanup."""
    with h:
        h.submit()
        h.tick()
        h.manager.begin_shutdown(h.now + 5)
        assert h.backend.shutdown_started.wait(1)
        stop = time.monotonic() + 1
        while len(h.backend.shutdown_threads) < 3 and time.monotonic() < stop:
            h.tick()
            time.sleep(0.005)
        assert len(h.backend.shutdown_threads) >= 3
        assert len(set(h.backend.shutdown_threads)) == 1
        assert h.backend.shutdown_threads[0] != threading.get_ident()
        assert not h.backend.finalizations
        h.backend.labels = frozenset()
        h.close()
        assert h.backend.finalizations == [threading.get_ident()]
        assert h.manager.shutdown_status().complete


def test_shutdown_never_blocks_owner_progress(h: _Harness) -> None:
    """A blocked SDK shutdown cannot block independent Lifecycle or owner rounds."""
    with h:
        h.submit()
        h.tick()
        h.backend.shutdown_gate = threading.Event()
        h.manager.begin_shutdown(h.now + 5)
        assert h.backend.shutdown_started.wait(1)
        start = time.monotonic()
        h.tick()
        assert time.monotonic() - start < 0.1
        assert not h.manager.shutdown_status().complete


def test_shutdown_handoff_keeps_independent_binding_and_deadlines_progressing(h: _Harness) -> None:
    """No owner-side Attempt call competes with a blocked sibling shutdown."""
    with h:
        other = _Backend()
        h.bind("other", other, domain_id="other-lender")
        lease, _ = h.submit()
        h.tick()
        h.backend.shutdown_gate = threading.Event()
        h.manager.stop_lender_domain("lender", "lender failed")
        assert h.backend.shutdown_started.wait(1)
        independent = _Lease(h.view)
        result = h.manager.submit(
            independent, replace(h.intent, binding_id="other"), write_return=_return_write
        )
        assert isinstance(result, Accepted)
        h.tick()
        other.attempts[0].outcome = Outcome.for_view(h.view, h.view.row_masks(True))
        other.attempts[0].ended = True
        start = time.monotonic()
        h.tick()
        assert time.monotonic() - start < 0.1
        assert not independent.released and not lease.released
        h.manager.authorize_return(result.operation_id, ReturnDecision(True, True))
        h.tick()
        assert independent.released and not lease.released
        assert not h.backend.call_violations
        h.now = 15
        h.tick()
        assert h.fatals and not lease.released


def test_shutdown_observes_final_outcome_after_early_lease_release(h: _Harness) -> None:
    """Empty labels cannot retire an Attempt whose final result is still unobserved."""
    with h:
        lease, operation = h.submit(direction=Direction.PUBLISH)
        h.tick()
        h.backend.attempts[0].ended = True
        h.tick()
        assert lease.released and operation in h.manager.operations
        h.backend.labels = frozenset()
        h.manager.begin_shutdown(h.now + 5)
        updates = []
        stop = time.monotonic() + 1
        while operation in h.manager.operations and time.monotonic() < stop:
            updates.extend(h.tick())
            h.wake.wait(0.001)
            h.wake.clear()
        final = next(
            update for update in updates if update.operation_id == operation and update.retired
        )
        assert final.outcome is h.backend.attempts[0].outcome
        assert final.outcome.fault == "shutdown"
        assert {tid for name, tid in h.backend.calls if name == "poll"} == {
            threading.get_ident(),
            h.backend.shutdown_owner,
        }


def test_lender_failure_stops_sibling_bindings_without_stopping_release(h: _Harness) -> None:
    """The domain stop precedes any sibling poll/return and preserves pending releases."""
    with h:
        sibling = _Backend()
        h.bind("sibling", sibling)
        exposed, exposed_id = h.submit()
        h.tick()
        failed, _ = h.submit()
        failed.poll_error = RuntimeError("lender state failed")
        pending = _Lease(h.view, ready=False)
        accepted = h.manager.submit(
            pending, replace(h.intent, binding_id="sibling"), write_return=_return_write
        )
        assert isinstance(accepted, Accepted)
        h.tick()
        polls = exposed.polls
        assert pending.polls == 0 and pending.released and failed.released
        assert not exposed.released and not exposed.returns
        assert all(not item.accepting for item in h.manager.bindings.values())
        untouched = _Lease(h.view)
        assert isinstance(
            h.manager.submit(untouched, h.intent, write_return=_return_write), Deferred
        )
        assert untouched.polls == 0 and not untouched.released
        h.backend.attempts[0].ended = True
        stop = time.monotonic() + 1
        updates = []
        while exposed_id in h.manager.operations and time.monotonic() < stop:
            updates.extend(h.tick())
            h.wake.wait(0.001)
            h.wake.clear()
        assert exposed.released and exposed.polls == polls and not exposed.returns
        final = next(
            update for update in updates if update.operation_id == exposed_id and update.retired
        )
        assert final.decision is ConsumptionDecision.DISCARD
        h.manager.stop_lender_domain("lender", "duplicate")
        assert len(h.manager.take_domain_failures()) == 1


def test_cumulative_shutdown_snapshot_preserves_proof_before_later_error(h: _Harness) -> None:
    """Coalescing keeps the first final result, empty access proof and later fault."""
    with h:
        lease, operation = h.submit()
        h.tick()
        original = h.backend.shutdown

        def fail_after_first(timeout: float) -> frozenset:
            """Report a later provider failure without invalidating earlier ended access."""
            labels = original(timeout)
            if len(h.backend.shutdown_threads) > 1:
                raise RuntimeError("late shutdown failure")
            return labels

        h.backend.shutdown = fail_after_first
        h.backend.labels = frozenset()
        h.manager.begin_shutdown(h.now + 5)
        stop = time.monotonic() + 1
        while len(h.backend.shutdown_threads) < 3 and time.monotonic() < stop:
            time.sleep(0.001)
        assert len(h.backend.shutdown_threads) >= 3
        updates = h.tick()
        assert lease.released and operation not in h.manager.operations
        assert updates[0].outcome is h.backend.attempts[0].outcome
        binding = h.manager.bindings["store"]
        assert binding.retained_labels == frozenset()
        assert "late shutdown failure" in binding.fault


def test_finalizer_queue_is_bounded_and_callback_failure_is_retained(h: _Harness) -> None:
    """False preserves provider ownership; accepted callbacks execute at most once."""
    with h:
        binding = h.manager.bindings["store"]
        calls = []
        for i in range(128):
            assert binding.cleanup.post(lambda i=i: calls.append(i))
        assert not binding.cleanup.post(lambda: calls.append("refused"))
        h.tick()
        assert calls == list(range(32))
        for _ in range(3):
            h.tick()
        assert calls == list(range(128))

        def failing() -> None:
            """Model a partially executed owner finalizer."""
            calls.append("failed")
            raise RuntimeError("hold release uncertain")

        assert binding.cleanup.post(failing)
        h.tick()
        h.tick()
        assert calls.count("failed") == 1 and binding.cleanup.fault is not None
        assert failing in binding.cleanup._failed and not binding.finalized


def test_binding_fault_does_not_block_unrelated_binding_before_fatal(h: _Harness) -> None:
    """Admission closure itself does not invent global or lease-level confinement."""
    with h:
        second = _Backend()
        h.bind("second", second)
        h.backend.error = RuntimeError("uncertain")
        h.submit()
        h.tick()
        result = h.manager.submit(
            _Lease(h.view), replace(h.intent, binding_id="second"), write_return=_return_write
        )
        assert isinstance(result, Accepted)
        h.tick()
        assert len(second.attempts) == 1


def test_unimplemented_scoped_policy_is_not_silently_downgraded(h: _Harness) -> None:
    """Lifecycle follow-up is required before selecting qualified scoped quarantine."""
    with h:
        with pytest.raises(ValueError, match="not implemented"):
            LifecycleBinding(h.watchdog, 5, "qualified-lease-isolation")


def test_empty_access_labels_do_not_seal_late_provider_cleanup(h: _Harness) -> None:
    """An empty access result and queue cannot hide cleanup not yet posted."""
    with h:
        original_post = h.backend.post
        allow_post = threading.Event()
        refused = threading.Event()

        def delayed_post(finalize: Callable[[], None]) -> bool:
            """Keep provider ownership until independently enabled finalizer posting."""
            if not allow_post.is_set():
                refused.set()
                return False
            return original_post(finalize)

        h.backend.post = delayed_post
        h.backend.labels = frozenset()
        h.manager.begin_shutdown(h.now + 5)
        assert refused.wait(1)
        h.tick()
        binding = h.manager.bindings["store"]
        assert binding.retained_labels == frozenset()
        assert not binding.cleanup.pending()
        assert not binding.finalized and not h.manager.shutdown_status().complete
        allow_post.set()
        h.close()
        assert h.backend.finalizations == [threading.get_ident()]
        assert h.manager.shutdown_status().complete


def test_backpressured_provider_finalizer_is_retried_without_false_completion(h: _Harness) -> None:
    """Queue refusal retains provider work even after all registrations have ended."""
    with h:
        binding = h.manager.bindings["store"]
        calls = []
        for i in range(128):
            assert binding.cleanup.post(lambda i=i: calls.append(i))
        refused = threading.Event()
        original_post = h.backend.post

        def observe_backpressure(finalize: Callable[[], None]) -> bool:
            """Expose the actual dispatcher refusal before the owner drains it."""
            accepted = original_post(finalize)
            if not accepted:
                refused.set()
            return accepted

        h.backend.post = observe_backpressure
        h.backend.labels = frozenset()
        h.manager.begin_shutdown(h.now + 5)
        assert refused.wait(1)
        h.tick()
        assert not h.manager.shutdown_status().complete
        h.close()
        assert calls == list(range(128))
        assert h.backend.finalizations == [threading.get_ident()]
        assert h.manager.shutdown_status().complete


def test_constructor_without_backend_is_uncertain_even_when_cleanup_is_complete(
    h: _Harness,
) -> None:
    """A private cleanup predicate cannot replace missing constructor access evidence."""
    with h:

        def factory(*args: object) -> None:
            """Violate Backend construction after a possibly escaped side effect."""
            return None

        with pytest.raises(TypeError, match="no Backend"):
            h.manager.assemble(
                "invalid",
                h.namespace,
                "m1",
                factory,
                domain_id="lender",
                lifecycle=h.policy,
                provider_cleanup_complete=lambda: True,
            )
        h.tick()
        binding = h.manager.bindings["invalid"]
        assert binding.construction_uncertain and binding.factory is factory
        assert not binding.finalized


def test_preexposure_lender_error_releases_failed_lease_and_drains_provider(h: _Harness) -> None:
    """An unexposed lease can be returned after a poll error without another poll."""
    with h:
        lease, operation = h.submit()
        lease.poll_error = RuntimeError("local copy state unknown")
        h.backend.labels = frozenset()
        h.tick()
        assert h.backend.shutdown_started.wait(1)
        binding = h.manager.bindings["store"]
        stop = time.monotonic() + 1
        while not binding.finalized and time.monotonic() < stop:
            h.tick()
            time.sleep(0.001)
        assert binding.retained_labels == frozenset()
        assert h.backend.posted and h.backend.finalizations
        binding.shutdown_thread.join(timeout=1)
        assert not binding.shutdown_thread.is_alive()
        assert binding.finalized and lease.released
        assert operation not in h.manager.operations
        assert lease.polls == 1 and not lease.returns
        h.now = 15.0
        h.watchdog.progress()
        h.tick()
        assert not h.fatals


def test_shutdown_keeps_binding_owner_before_pending_lease_release_can_fail(h: _Harness) -> None:
    """Provider cleanup observed first cannot close the owner of a later failed release."""
    with h:
        lease, operation = h.submit(ready=False)
        lease.release_error = RuntimeError("pending lease release failed")
        h.backend.labels = frozenset()
        h.manager.begin_shutdown(h.now + 5)
        stop = time.monotonic() + 1
        while not h.backend.posted and time.monotonic() < stop:
            time.sleep(0.001)
        assert h.backend.posted
        h.tick()
        binding = h.manager.bindings["store"]
        assert binding.retained_labels == frozenset() and h.backend.finalizations
        assert h.manager.operations[operation].release_failed and not binding.finalized
        h.now = 15.0
        h.watchdog.progress()
        assert h.fatals and operation in h.manager.operations and not lease.released
