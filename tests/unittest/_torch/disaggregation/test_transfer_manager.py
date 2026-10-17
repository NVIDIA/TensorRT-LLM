# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Holder contracts; doubles isolate only external I/O and lender-owned CUDA copies."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

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
    Deferred,
    Direction,
    LifecycleBinding,
    Marking,
    Rejected,
    Submission,
    TransferIntent,
    TransferManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LayerGroupBlocks,
    LeaseView,
    MemoryRegion,
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
    marks: list[tuple[np.ndarray, ...]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    poll_error: Exception | None = None
    mark_error: Exception | None = None
    release_error: Exception | None = None

    def poll(self) -> LeaseView | None:
        """Return the ready protected view, detecting invalid polling after release."""
        assert not self.released
        self.polls += 1
        if self.poll_error is not None:
            raise self.poll_error
        return self.view if self.ready and self.failure is None else None

    def mark_arrived(self, masks: tuple[np.ndarray, ...]) -> None:
        """Record a one-shot decision, including marking after safe lease release."""
        assert not self.marks
        self.marks.append(tuple(mask.copy() for mask in masks))
        self.actions.append("mark")
        if self.mark_error is not None:
            raise self.mark_error

    def release(self) -> None:
        """Record protection return without claiming local copies have completed."""
        assert not self.released
        self.actions.append("release")
        if self.release_error is not None:
            raise self.release_error
        self.released = True


@dataclass
class _Attempt:
    """Independent result and access-end observations, without physical progress."""

    outcome: Outcome | None = None
    ended: bool = False
    cancels: int = 0
    before_poll: Callable[[], None] | None = None

    def poll(self) -> Outcome | None:
        """Return the fixed result, optionally crossing a deterministic deadline."""
        if self.before_poll is not None:
            self.before_poll()
        return self.outcome

    def quiesced(self) -> bool:
        """Return independent whole-Attempt access evidence."""
        return self.ended

    def cancel(self) -> None:
        """Count advisory cancellation without inventing completion."""
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

    def _submit(self, view: LeaseView, hint: object, direction: str) -> _Attempt:
        """Return a fixed handle or inject an explicit submission disposition."""
        self.submissions.append((view, hint, direction))
        if self.error is not None:
            raise self.error
        attempt = _Attempt()
        self.attempts.append(attempt)
        return attempt

    def fetch(self, view: LeaseView, source: object) -> _Attempt:
        """Accept destination writes without completing them."""
        return self._submit(view, source, "fetch")

    def publish(self, view: LeaseView, destination: object) -> _Attempt:
        """Accept source reads without requiring an inference request."""
        return self._submit(view, destination, "publish")

    def shutdown(self, timeout: float) -> frozenset:
        """Force fixed Outcomes but keep access evidence independently controlled."""
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

    def __init__(self) -> None:
        """Construct the explicitly known HOST fixture without claiming activation."""
        self.now = 10.0
        self.fatals: list[object] = []
        self.watchdog = RetirementWatchdog(self.fatals.append, clock=lambda: self.now)
        self.wake = threading.Event()
        self.manager = TransferManager(wake=self.wake.set, clock=lambda: self.now)
        self.namespace = RegionNamespace((MemoryRegion("layout", 4096, 64, 16, 4, (7,)),), "HOST")
        self.view = LeaseView(
            (
                LayerGroupBlocks(
                    7,
                    np.array([2, 3], dtype=np.int64),
                    np.zeros((2, 54), dtype=np.uint8),
                    np.array([4096, 4128], dtype=np.int64),
                    0,
                ),
            )
        )
        self.backend = _Backend()
        self.policy = LifecycleBinding(self.watchdog, 5)
        self.bind("store", self.backend)
        self.intent = TransferIntent("store", "m1", Direction.FETCH, "HOST")

    def bind(self, name: str, backend: _Backend) -> None:
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
            lifecycle=self.policy,
            provider_cleanup_complete=lambda: backend.posted,
        )

    def submit(
        self, *, direction: Direction = Direction.FETCH, ready: bool = True
    ) -> tuple[_Lease, int]:
        """Acquire externally and hand off without any Request/lender dependency."""
        lease = _Lease(self.view, ready=ready)
        result = self.manager.submit(lease, replace(self.intent, direction=direction))
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
        ("memory_kind", "DEVICE"),
        ("direction", "invalid"),
    ],
)
def test_rejected_intent_leaves_lease_untouched(h: _Harness, field: str, value: object) -> None:
    """Metadata mismatch never acquires lease ownership or substitutes a source."""
    with h:
        lease = _Lease(h.view)
        result = h.manager.submit(lease, replace(h.intent, **{field: value}))
        assert isinstance(result, Rejected)
        assert not h.manager.operations and lease.polls == 0 and not lease.released


def test_deferred_binding_leaves_caller_ownership(h: _Harness) -> None:
    """Admission refusal must not poll, mark or return the caller's lease."""
    with h:
        h.manager.bindings["store"].accepting = False
        lease = _Lease(h.view)
        assert isinstance(h.manager.submit(lease, h.intent), Deferred)
        assert lease.polls == 0 and not lease.actions and not h.manager.operations


def test_same_lease_cannot_have_two_accepted_operations(h: _Harness) -> None:
    """M1 protects one Lease/Attempt owner and rejects accidental active reuse."""
    with h:
        lease, _ = h.submit()
        assert isinstance(h.manager.submit(lease, h.intent), Rejected)
        assert len(h.manager.operations) == 1


@pytest.mark.parametrize("outcome_first", [False, True])
def test_delivery_and_access_end_are_independent(h: _Harness, outcome_first: bool) -> None:
    """Both evidence orders keep one mark/release and no request-ready decision."""
    with h:
        lease, operation = h.submit()
        h.tick()
        attempt = h.backend.attempts[0]
        if outcome_first:
            h.complete()
            (update,) = h.tick()
            assert update.marking is Marking.DELIVERED and not update.access_ended
            assert not lease.released
            attempt.ended = True
        else:
            attempt.ended = True
            (update,) = h.tick()
            assert update.access_ended and update.outcome is None
            assert lease.released and not lease.marks
            assert operation in h.manager.operations
            h.complete()
        (update,) = h.tick()
        assert update.retired and update.outcome is attempt.outcome
        assert lease.released and lease.marks[0][0].all()
        assert h.tick() == () and not h.manager.operations


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("abandoned", [False, True])
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
        assert not lease.marks if direction is Direction.PUBLISH else len(lease.marks) == 1


def test_unseen_abandonment_needs_no_mark_or_submission(h: _Harness) -> None:
    """An unpolled accepted lease can be abandoned without inventing a view."""
    with h:
        lease, operation = h.submit(ready=False)
        h.manager.abandon(operation, "request cancelled")
        (update,) = h.tick()
        assert update.retired and update.marking is Marking.UNDECIDED
        assert lease.released and lease.polls == 0 and not lease.marks
        assert not h.backend.submissions


@pytest.mark.parametrize("supported", [False, True])
def test_cancellation_is_advisory_and_one_shot(h: _Harness, supported: bool) -> None:
    """Cancel neither proves access end nor changes a fixed consumption decision."""
    with h:
        h.backend.capabilities = Capabilities(True, True, supported)
        lease, operation = h.submit()
        h.tick()
        h.manager.abandon(operation, "acquisition abandoned")
        h.manager.abandon(operation, "duplicate")
        h.tick()
        assert h.backend.attempts[0].cancels == int(supported)
        assert not lease.released and len(lease.marks) == 1 and not lease.marks[0][0].any()


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
        assert not lease.marks[0][0].any()
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
        assert not lease.marks[0][0].any()
        assert h.manager.operations[operation].outcome is None


def test_timeout_committed_inside_poll_wins_before_marking(h: _Harness) -> None:
    """A poll crossing the exact exposure deadline cannot consume late rows."""
    with h:
        lease, operation = h.submit()
        h.tick()
        h.complete()
        h.backend.attempts[0].before_poll = lambda: setattr(h, "now", 15.0)
        (update,) = h.tick()
        assert update.marking is Marking.ABANDONED
        assert not lease.marks[0][0].any()
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
        assert not lease.marks


@pytest.mark.parametrize("method", ["poll", "mark", "release"])
def test_unknown_lease_failure_retains_owner_and_stops_binding(h: _Harness, method: str) -> None:
    """Ambiguous manager-side errors are not converted into safe resource return."""
    with h:
        lease, operation = h.submit()
        setattr(lease, method + "_error", RuntimeError("manager failed"))
        h.tick()
        if method != "poll":
            h.complete()
            h.backend.attempts[0].ended = True
            h.tick()
        assert not lease.released and operation in h.manager.operations
        assert not h.manager.bindings["store"].accepting
        h.tick()
        assert lease.actions.count(method) <= 1


def test_documented_lease_failure_before_view_can_retire(h: _Harness) -> None:
    """A canonical failed lease has no backend exposure and needs no marking."""
    with h:
        lease, _ = h.submit()
        lease.failure = "staging capacity unavailable"
        (update,) = h.tick()
        assert update.retired and update.fault == lease.failure
        assert lease.released and not lease.marks


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

        def factory(*args: object) -> None:
            """Reject before any resource or callback escapes."""
            raise FactoryRejected("nothing retained")

        with pytest.raises(FactoryRejected):
            h.manager.assemble(
                "rejected",
                h.namespace,
                "m1",
                factory,
                lifecycle=h.policy,
                provider_cleanup_complete=lambda: True,
            )
        assert "rejected" not in h.manager.bindings


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
        h.backend.shutdown_gate = threading.Event()
        h.manager.begin_shutdown(h.now + 5)
        assert h.backend.shutdown_started.wait(1)
        start = time.monotonic()
        h.tick()
        assert time.monotonic() - start < 0.1
        assert not h.manager.shutdown_status().complete


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
        result = h.manager.submit(_Lease(h.view), replace(h.intent, binding_id="second"))
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
                lifecycle=h.policy,
                provider_cleanup_complete=lambda: True,
            )
        h.tick()
        binding = h.manager.bindings["invalid"]
        assert binding.construction_uncertain and binding.factory is factory
        assert not binding.finalized


def test_preexposure_unsafe_lease_keeps_lifecycle_after_provider_cleanup(h: _Harness) -> None:
    """Ended provider access must not erase the only owner of a failed lender call."""
    with h:
        lease, operation = h.submit()
        lease.poll_error = RuntimeError("local copy state unknown")
        h.backend.labels = frozenset()
        h.tick()
        assert h.backend.shutdown_started.wait(1)
        binding = h.manager.bindings["store"]
        stop = time.monotonic() + 1
        while not h.backend.finalizations and time.monotonic() < stop:
            h.tick()
            time.sleep(0.001)
        assert binding.retained_labels == frozenset()
        assert h.backend.posted and h.backend.finalizations
        binding.shutdown_thread.join(timeout=1)
        assert not binding.shutdown_thread.is_alive()
        assert binding.lifecycle is not None and not binding.finalized
        assert not h.manager.shutdown_status().complete
        h.now = 15.0
        h.watchdog.progress()
        h.tick()
        assert h.fatals and not lease.released
        assert operation in h.manager.operations and binding.lifecycle is not None


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
        assert h.manager.operations[operation].unsafe_lease and not binding.finalized
        h.now = 15.0
        h.watchdog.progress()
        assert h.fatals and operation in h.manager.operations and not lease.released
