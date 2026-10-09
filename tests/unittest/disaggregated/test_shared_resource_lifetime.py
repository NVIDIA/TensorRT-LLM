# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU ownership tests with external provider, lender, and CUDA evidence doubles.

Production adapter mapping and lifecycle arbitration run unchanged. These tests
cannot qualify real CUDA copies, page movement, or a provider's no-access proof.
"""

from __future__ import annotations

import gc
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.shared import (
    Attempt,
    CacheExtent,
    CancelDisposition,
    Cancelled,
    Delivered,
    Failed,
    Outcome,
    Route,
    SubmissionRejected,
)
from tensorrt_llm._torch.disaggregation.lifecycle.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.resource.shared import (
    SHARED_CONTRACT_REVISION,
    SharedRuntimeProfile,
)
from tensorrt_llm._torch.disaggregation.resource.shared_lifetime import (
    SharedLeaseOperation,
    SharedStagingAdapter,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import GroupRun, Part, RegionView

pytestmark = pytest.mark.cpu_only
_NAMES = (bytes([1]) * 54, bytes([2]) * 54)


class _Lease:
    """Externally controlled lender boundary with thread/cleanup observations."""

    def __init__(self) -> None:
        """Create two staging rows and no completed arrival or release."""
        self.view = RegionView(
            (
                GroupRun(
                    0,
                    np.array([0, 1]),
                    np.array([list(name) for name in _NAMES], dtype=np.uint8),
                    np.array([1000, 1016]),
                    0,
                ),
            )
        )
        self.ready = True
        self.failure: str | None = None
        self.marks: list[tuple[np.ndarray, ...]] = []
        self.release_count = 0
        self.threads: list[int] = []
        self.mark_error: Exception | None = None

    def poll(self) -> RegionView | None:
        """Return the fixture's supplied readiness without simulating copies."""
        self.threads.append(threading.get_ident())
        return self.view if self.ready else None

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """Record the whole-unit masks and optionally simulate a copy error.

        Args:
            masks: One boolean array per public lender run.
        """
        self.threads.append(threading.get_ident())
        self.marks.append(tuple(mask.copy() for mask in masks))
        if self.mark_error is not None:
            raise self.mark_error

    def release(self) -> None:
        """Count calls so duplicate adapter cleanup cannot hide behind a fake."""
        self.threads.append(threading.get_ident())
        self.release_count += 1


class _Registration:
    """Provider handle with observable retryable deregistration failure."""

    def __init__(self, events: list[str]) -> None:
        """Retain the fixture log and initially allow closure.

        Args:
            events: Shared log proving release ordering.
        """
        self.events = events
        self.fail = False
        self.calls = 0

    def close(self) -> None:
        """Record closure and raise while the fixture keeps it failing."""
        self.calls += 1
        self.events.append("deregister")
        if self.fail:
            raise RuntimeError("registration remains open")


class _Attempt:
    """Externally selected outcome, independent of the access-end proof."""

    def __init__(self) -> None:
        """Create a pending logical outcome."""
        self.outcome: Outcome | None = None

    def poll(self) -> Outcome | None:
        """Return the fixture's current logical answer."""
        return self.outcome


class _CancellableAttempt(_Attempt):
    """Record the adapter's request independently of outcome and access evidence."""

    def __init__(self) -> None:
        """Create a pending attempt with observable cancellation behavior."""
        super().__init__()
        self.cancel_calls = 0
        self.cancel_threads: list[int] = []
        self.on_cancel: Callable[[], None] | None = None
        self.cancel_error: Exception | None = None

    def request_cancel(self) -> CancelDisposition:
        """Record a best-effort request without changing outcome or access.

        Returns:
            REQUESTED unless the test injects a provider exception.

        Raises:
            Exception: The injected provider error.
        """
        self.cancel_calls += 1
        self.cancel_threads.append(threading.get_ident())
        if self.on_cancel is not None:
            self.on_cancel()
        if self.cancel_error is not None:
            raise self.cancel_error
        return CancelDisposition.REQUESTED


class _Backend:
    """Controllable external calls, without copying any adapter ownership logic."""

    def __init__(self, events: list[str]) -> None:
        """Initialize a provider with blocked physical completion.

        Args:
            events: Shared log recording backend/hold ordering.
        """
        self.events = events
        self.attempt = _Attempt()
        self.submission_error: Exception | None = None
        self.quiescence: object = True
        self.entered = threading.Event()
        self.allow = threading.Event()
        self.returned = threading.Event()
        self.registrations: list[_Registration] = []
        self.spans: list[tuple[int, int]] = []
        self.register_fail_at: int | None = None
        self.submissions = 0
        self.wait_calls = 0
        self.wait_threads: list[threading.Thread] = []
        self.extents: list[CacheExtent] = []

    def register_pool(self, address: int, size: int) -> _Registration:
        """Register a physical span, optionally failing before registration.

        Args:
            address: Base address of the exact registered allocation.
            size: Number of accessible bytes.

        Returns:
            An independently closable provider handle.
        """
        self.events.append("register")
        if self.register_fail_at == len(self.registrations):
            raise RuntimeError("register failed")
        self.spans.append((address, size))
        handle = _Registration(self.events)
        self.registrations.append(handle)
        return handle

    def fetch(self, extent: CacheExtent, *, route: Route | None = None) -> Attempt:
        """Record submission and return its independently controlled Attempt.

        Args:
            extent: Real adapter-mapped units.
            route: Unused route at this external test boundary.

        Returns:
            The fixture Attempt, unless an injected submission error raises.
        """
        self.submissions += 1
        self.extents.append(extent)
        if self.submission_error is not None:
            raise self.submission_error
        return self.attempt

    def publish(self, extent: CacheExtent) -> Attempt:
        """Use the same controllable submission boundary for a read lease.

        Args:
            extent: Source units mapped by the production adapter.

        Returns:
            The fixture's independent logical Attempt.
        """
        return self.fetch(extent)

    def quiesce(self, attempts: Iterable[Attempt]) -> bool:
        """Block until released, then supply the selected external proof.

        Args:
            attempts: Exactly the retained Attempt passed by the adapter.

        Returns:
            The selected truth value; invalid truthy evidence is also tested.
        """
        assert tuple(attempts) == (self.attempt,)
        self.wait_calls += 1
        self.wait_threads.append(threading.current_thread())
        self.entered.set()
        assert self.allow.wait(2), "test did not unblock the backend proof"
        try:
            if isinstance(self.quiescence, Exception):
                raise self.quiescence
            return self.quiescence
        finally:
            self.returned.set()

    def settle(self, attempts: Iterable[Attempt]) -> None:
        """Reject use of the provider's unbounded logical wait.

        Args:
            attempts: Handles that must instead be polled nonblockingly.
        """
        pytest.fail("the owner thread must not call blocking settle")

    def probe(self, name: bytes, units: Sequence[bytes]) -> frozenset[bytes] | None:
        """Expose the required fetch capability without participating in tests.

        Args:
            name: Content namespace.
            units: Proposed unit names.

        Returns:
            None because the fake offers no advisory lookup.
        """
        return None

    def open_route(self, hint: Mapping[str, object]) -> Route:
        """Reject routing because this fixture models one source.

        Args:
            hint: Unused source hint.

        Raises:
            NotImplementedError: The fixture has only one source.
        """
        raise NotImplementedError


class _Fixture:
    """Wire real lifecycle/adapter code to externally controlled dependencies."""

    def __init__(self) -> None:
        """Create a registered adapter without exposing a transfer operation."""
        self.events: list[str] = []
        self.clock = Mock(return_value=10.0)
        self.watchdog = RetirementWatchdog(Mock(), clock=self.clock)
        self.registration_retirement = self.watchdog.create_owner(-1, "receive", 5)
        self.manager = Mock()
        self.hold = Mock()
        self.hold.release.side_effect = lambda: self.events.append("release hold")
        self.lender = Mock()
        self.lender.parts = (Part("layout", 1000, 64, 16, 4),)
        self.lender.hold_parts.side_effect = self.take_hold
        self.backend = _Backend(self.events)
        self.copy_done = Mock(return_value=True)
        self.record_copy = Mock(return_value=self.copy_done)
        self.profile = SharedRuntimeProfile(
            contract_revision=SHARED_CONTRACT_REVISION,
            backend_revision="fixture-backend",
            backend_kind="native_nixl",
            manager_kind="KVCacheManagerV2",
            cache_dtype="bfloat16",
            attention_backend="TRTLLM",
            attention_kind="mha",
            layout="HND",
            parallelism=(1, 1, 1, 1),
            staging="manager_host",
            committed_whole_blocks=True,
            extra_features=frozenset(),
        )
        self.adapter = SharedStagingAdapter(
            self.manager,
            self.lender,
            self.profile,
            self.backend,
            self.registration_retirement,
            self.record_copy,
        )
        self.adapter.register()
        self.lease = _Lease()
        self.operation: SharedLeaseOperation | None = None

    def take_hold(self) -> Mock:
        """Log the hold acquisition before returning the external hold double.

        Returns:
            The fixture's observable parts hold.
        """
        self.events.append("hold")
        return self.hold

    def submit(self, writing: bool = True) -> SharedLeaseOperation:
        """Submit a ready lease through a real, uniquely owned deadline.

        Args:
            writing: Select fetch or publication.

        Returns:
            The production operation binding.
        """
        retirement = self.watchdog.create_owner(1, "receive" if writing else "send", 5)
        submit = self.adapter.submit_fetch if writing else self.adapter.submit_publish
        self.operation = submit(self.lease, self.backend, retirement)
        return self.operation

    def finish_proof(self) -> None:
        """Release the external backend wait and join its owned proof workers."""
        self.backend.allow.set()
        assert self.backend.returned.wait(1)
        assert self.operation is not None
        proof = self.operation._proof
        assert proof is not None
        try:
            proof.result(timeout=1)
        except RuntimeError:
            pass
        for worker in self.backend.wait_threads:
            worker.join(timeout=1)
            assert not worker.is_alive()


@pytest.fixture
def bound() -> Iterable[_Fixture]:
    """Yield fixture ownership and always unblock its daemon backend thread.

    Yields:
        A registered production adapter with controllable boundaries.
    """
    fixture = _Fixture()
    try:
        yield fixture
    finally:
        fixture.backend.allow.set()


@pytest.mark.parametrize("writing", [False, True], ids=["publish", "fetch"])
@pytest.mark.parametrize("outcome_first", [False, True], ids=["access-first", "outcome-first"])
def test_outcome_and_access_end_are_independent(
    bound: _Fixture, writing: bool, outcome_first: bool
) -> None:
    """Both evidence orders retain leases and invoke lender methods only on owner."""
    operation = bound.submit(writing)
    assert bound.backend.entered.wait(1)
    if outcome_first:
        bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    else:
        bound.finish_proof()
    bound.adapter.progress()
    assert not operation.released
    assert not bound.lease.marks
    if outcome_first:
        assert isinstance(operation.outcome, Delivered)
        bound.finish_proof()
    else:
        assert operation.outcome is None
        bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.adapter.progress()
    assert operation.released
    assert bound.lease.release_count == 1
    assert len(bound.lease.marks) == int(writing)
    assert set(bound.lease.threads) == {threading.get_ident()}
    bound.adapter.progress()
    assert bound.adapter.close() and bound.adapter.close()
    assert bound.lease.release_count == 1
    assert bound.events == ["hold", "register", "deregister", "release hold"]
    bound.watchdog.require_retired()


@pytest.mark.parametrize("served", [frozenset(), frozenset(_NAMES[:1])], ids=["miss", "partial"])
def test_only_complete_served_units_are_marked(bound: _Fixture, served: frozenset[bytes]) -> None:
    """A miss and a partial delivery preserve the exact public row-mask mapping."""
    operation = bound.submit()
    bound.backend.attempt.outcome = Delivered(served)
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.released
    assert bound.lease.marks[0][0].tolist() == [name in served for name in _NAMES]


def test_copy_barrier_survives_request_exit_and_blocks_shutdown(bound: _Fixture) -> None:
    """Local copy evidence, independent of request liveness, gates pool release."""
    bound.copy_done.return_value = False
    operation = bound.submit()
    manager_ref = weakref.ref(bound.manager)
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.finish_proof()
    bound.adapter.progress()
    assert len(bound.lease.marks) == 1
    assert not operation.released
    del bound.manager
    gc.collect()
    assert manager_ref() is not None
    assert not bound.adapter.close()
    assert bound.backend.registrations[0].calls == 0
    bound.copy_done.return_value = True
    assert bound.adapter.close()
    assert operation.released
    assert len(bound.lease.marks) == 1
    assert bound.events[-2:] == ["deregister", "release hold"]


@pytest.mark.parametrize("terminal", [Failed("failed"), Cancelled(False)])
def test_terminal_outcome_is_stable_during_late_completion(
    bound: _Fixture, terminal: Outcome
) -> None:
    """Failure/cancellation cannot turn into success when a late attempt finishes."""
    operation = bound.submit()
    if isinstance(terminal, Cancelled):
        bound.adapter.cancel(operation)
    else:
        bound.backend.attempt.outcome = terminal
        bound.adapter.progress()
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.outcome == terminal
    assert operation.released
    assert bound.lease.marks[0][0].tolist() == [False, False]


def test_unsupported_cancellation_keeps_logical_outcome_and_physical_roots(bound: _Fixture) -> None:
    """A provider without cancellation still supports local cancellation and drain."""
    operation = bound.submit()
    bound.adapter.cancel(operation, by_peer=True)
    assert operation.outcome == Cancelled(True)
    assert operation.cancel_disposition is CancelDisposition.UNSUPPORTED
    assert operation.cancel_error is None
    assert not bound.adapter.close()
    assert operation.outcome == Cancelled(True)
    assert not operation.released
    assert bound.backend.registrations[0].calls == 0
    bound.hold.release.assert_not_called()
    bound.finish_proof()
    assert bound.adapter.close()
    assert operation.released


def test_cancellation_commits_before_one_request_outside_arbiter_lock(bound: _Fixture) -> None:
    """An acknowledgement cannot release blocked I/O or pending local copies."""
    attempt = _CancellableAttempt()
    bound.backend.attempt = attempt
    operation = bound.submit()
    assert bound.backend.entered.wait(1)

    def observe_cancel() -> None:
        """Check logical visibility and independent arbiter progress at dispatch."""
        assert operation.outcome == Cancelled(False)
        acquired: list[bool] = []

        def check_lock() -> None:
            """Observe that cancellation does not hold the watchdog's arbiter."""
            acquired.append(operation.retirement.lock.acquire(blocking=False))
            if acquired[-1]:
                operation.retirement.lock.release()

        worker = threading.Thread(target=check_lock)
        worker.start()
        worker.join(timeout=1)
        assert acquired == [True]

    attempt.on_cancel = observe_cancel
    bound.adapter.cancel(operation)
    bound.adapter.cancel(operation, by_peer=True)
    bound.adapter.progress()
    assert not bound.adapter.close()
    assert not bound.adapter.close()
    assert attempt.cancel_calls == 1
    assert attempt.cancel_threads == [threading.get_ident()]
    assert operation.cancel_disposition is CancelDisposition.REQUESTED
    assert operation.cancel_error is None
    assert not operation.released
    assert bound.backend.registrations[0].calls == 0
    bound.hold.release.assert_not_called()
    bound.copy_done.return_value = False
    bound.finish_proof()
    bound.adapter.progress()
    assert not operation.released
    assert bound.lease.marks[0][0].tolist() == [False, False]
    bound.copy_done.return_value = True
    assert bound.adapter.close()
    bound.adapter.cancel(operation)
    assert attempt.cancel_calls == 1
    assert bound.lease.release_count == 1


@pytest.mark.parametrize("observed", [False, True], ids=["unobserved", "committed"])
def test_first_local_outcome_wins_completion_cancellation_race(
    bound: _Fixture, observed: bool
) -> None:
    """Backend success wins only after the adapter commits that logical outcome."""
    attempt = _CancellableAttempt()
    bound.backend.attempt = attempt
    operation = bound.submit()
    delivered = Delivered(frozenset(_NAMES))
    attempt.outcome = delivered
    if observed:
        bound.adapter.progress()
    bound.adapter.cancel(operation)
    expected = delivered if observed else Cancelled(False)
    assert operation.outcome == expected
    assert attempt.cancel_calls == int(not observed)
    assert attempt.outcome == delivered
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.released
    assert operation.outcome == expected
    assert bound.lease.marks[0][0].tolist() == [observed, observed]
    assert bound.adapter.close()


def test_cancellation_error_does_not_obstruct_other_operations_or_proof(bound: _Fixture) -> None:
    """Provider cancellation errors remain diagnostic while all operations progress."""
    attempt = _CancellableAttempt()
    attempt.cancel_error = RuntimeError("cancel submission failed")
    bound.backend.attempt = attempt
    first = bound.submit()
    assert bound.backend.entered.wait(1)
    attempt.outcome = Failed("provider delivery failed")
    bound.backend.attempt = _Attempt()
    second_lease = _Lease()
    second = bound.adapter.submit_fetch(
        second_lease, bound.backend, bound.watchdog.create_owner(2, "receive", 5)
    )
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.adapter.progress()
    assert first.outcome == Failed("provider delivery failed")
    assert first.cancel_error == "RuntimeError: cancel submission failed"
    assert first.cancel_disposition is None
    assert not first.released
    assert second.outcome == Delivered(frozenset(_NAMES))
    bound.operation = second
    bound.finish_proof()
    bound.adapter.progress()
    assert first.released and second.released
    assert bound.lease.release_count == second_lease.release_count == 1
    assert attempt.cancel_calls == 1
    assert bound.adapter.close()


def test_timeout_forwards_cancellation_only_during_owner_progress(bound: _Fixture) -> None:
    """Delayed request dispatch cannot extend the watchdog's fixed grace period."""
    attempt = _CancellableAttempt()
    bound.backend.attempt = attempt
    operation = bound.submit()
    assert bound.backend.entered.wait(1)
    bound.clock.return_value = 15.0
    worker = threading.Thread(target=bound.watchdog.progress)
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert operation.outcome == Failed("transfer timeout")
    assert attempt.cancel_calls == 0
    bound.clock.return_value = 18.0
    bound.adapter.progress()
    bound.adapter.cancel(operation)
    assert attempt.cancel_threads == [threading.get_ident()]
    assert operation.cancel_disposition is CancelDisposition.REQUESTED
    assert operation.outcome == Failed("transfer timeout")
    bound.clock.return_value = 19.999
    bound.watchdog.progress()
    assert bound.watchdog.fatal is None
    bound.clock.return_value = 20.0
    bound.watchdog.progress()
    assert bound.watchdog.fatal is not None
    assert bound.watchdog.fatal.started_at == 15.0
    assert bound.watchdog.fatal.deadline == 20.0
    bound.finish_proof()
    bound.adapter.progress()
    assert attempt.cancel_calls == 1
    assert not operation.released
    bound.hold.release.assert_not_called()


def test_cancellation_preserves_already_observed_quiescence(bound: _Fixture) -> None:
    """A conforming cancellation request cannot invalidate positive access evidence."""
    attempt = _CancellableAttempt()
    bound.backend.attempt = attempt
    operation = bound.submit()
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.outcome is None
    assert not operation.released
    bound.adapter.cancel(operation)
    bound.adapter.progress()
    assert operation.released
    assert operation.outcome == Cancelled(False)
    assert operation.cancel_disposition is CancelDisposition.REQUESTED
    assert bound.backend.wait_calls == 1
    assert bound.adapter.close()


def test_request_timeout_stays_failed_after_late_success(bound: _Fixture) -> None:
    """A timeout commits independently while quiesce is blocked in a background thread."""
    operation = bound.submit()
    assert bound.backend.entered.wait(1)
    bound.clock.return_value = 15.0
    bound.watchdog.progress()
    assert operation.outcome == Failed("transfer timeout")
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.outcome == Failed("transfer timeout")
    assert operation.released
    assert bound.lease.marks[0][0].tolist() == [False, False]


def test_blocked_quiesce_cannot_delay_watchdog_and_late_proof_cannot_release(
    bound: _Fixture,
) -> None:
    """Independent lifecycle containment wins at the exact fixed expiry boundary."""
    operation = bound.submit()
    assert bound.backend.entered.wait(1)
    bound.clock.return_value = 20.0
    bound.watchdog.progress()
    assert bound.watchdog.fatal is not None
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.finish_proof()
    bound.adapter.progress()
    assert not operation.released
    assert not bound.lease.marks
    assert not bound.adapter.close()
    bound.hold.release.assert_not_called()


@pytest.mark.parametrize("rejected", [True, False], ids=["rejected", "escaped"])
def test_submission_failure_preserves_access_uncertainty(bound: _Fixture, rejected: bool) -> None:
    """Only SubmissionRejected certifies that a provider submission never escaped."""
    bound.backend.submission_error = (
        SubmissionRejected("rejected") if rejected else RuntimeError("escaped")
    )
    operation = bound.submit()
    assert isinstance(operation.outcome, Failed)
    assert operation.released is rejected
    assert bound.backend.wait_calls == 0
    if not rejected:
        bound.clock.return_value = 15.0
        bound.watchdog.progress()
        assert bound.watchdog.fatal is not None
        assert not operation.released


@pytest.mark.parametrize("proof", [False, 1, RuntimeError("proof failed")])
def test_unproven_access_requires_explicit_retry(bound: _Fixture, proof: object) -> None:
    """False, truthy non-bool, and exceptional proofs fail closed until a later True."""
    bound.backend.quiescence = proof
    operation = bound.submit()
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.finish_proof()
    bound.adapter.progress()
    assert not operation.released
    assert not bound.lease.marks
    bound.adapter.progress()
    assert bound.backend.wait_calls == 1
    bound.backend.quiescence = True
    bound.backend.returned.clear()
    bound.adapter.retry_quiescence(operation)
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.released
    assert bound.backend.wait_calls == 2


def test_quiescence_worker_start_failure_allows_explicit_retry(
    bound: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed proof worker preserves the submitted operation and allows recovery."""
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("no thread")))
        operation = bound.submit()
    assert bound.backend.submissions == 1
    assert bound.backend.wait_calls == 0
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.adapter.progress()
    assert operation.outcome == Delivered(frozenset(_NAMES))
    assert not operation.released
    assert not bound.lease.marks
    assert bound.lease.release_count == 0
    bound.hold.release.assert_not_called()
    bound.adapter.retry_quiescence(operation)
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.released
    assert bound.backend.wait_calls == 1
    assert bound.lease.release_count == 1
    assert bound.adapter.close()
    bound.watchdog.require_retired()


@pytest.mark.parametrize("failure_site", ["mark", "record", "query"])
def test_copy_errors_retain_roots_and_do_not_repeat_marking(
    bound: _Fixture, failure_site: str
) -> None:
    """Any ambiguous local-copy boundary retains the loan through containment."""
    if failure_site == "mark":
        bound.lease.mark_error = RuntimeError("copy error")
    elif failure_site == "record":
        bound.record_copy.side_effect = RuntimeError("record error")
    else:
        bound.copy_done.side_effect = RuntimeError("query error")
    operation = bound.submit()
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.finish_proof()
    bound.adapter.progress()
    bound.adapter.progress()
    assert len(bound.lease.marks) == 1
    assert not operation.released
    bound.clock.return_value = 15.0
    bound.watchdog.progress()
    assert bound.watchdog.fatal is not None


def test_unknown_served_name_fails_without_copying_untrusted_rows(bound: _Fixture) -> None:
    """Backend unit claims outside the requested extent cannot reach manager pages."""
    operation = bound.submit()
    bound.backend.attempt.outcome = Delivered(frozenset({b"not requested"}))
    bound.finish_proof()
    bound.adapter.progress()
    assert isinstance(operation.outcome, Failed)
    assert operation.released
    assert bound.lease.marks[0][0].tolist() == [False, False]


def test_source_copy_not_ready_remains_owned_by_caller(bound: _Fixture) -> None:
    """Queueing/source readiness never starts the lifecycle transfer clock."""
    bound.lease.ready = False
    retirement = bound.watchdog.create_owner(1, "send", 5)
    with pytest.raises(ValueError, match="not ready"):
        bound.adapter.submit_publish(bound.lease, bound.backend, retirement)
    assert bound.backend.submissions == 0
    assert bound.lease.release_count == 0
    bound.clock.return_value = 100.0
    bound.watchdog.progress()
    assert bound.watchdog.fatal is None
    bound.lease.ready = True
    operation = bound.adapter.submit_publish(bound.lease, bound.backend, retirement)
    assert operation.outcome is None
    bound.operation = operation
    bound.finish_proof()
    assert bound.adapter.close()


def test_close_retries_same_registration_before_releasing_hold(bound: _Fixture) -> None:
    """Failed deregistration preserves its precise handle and allocation hold."""
    registration = bound.backend.registrations[0]
    registration.fail = True
    assert not bound.adapter.close()
    bound.hold.release.assert_not_called()
    registration.fail = False
    assert bound.adapter.close()
    assert registration.calls == 2
    assert bound.adapter.close()
    assert registration.calls == 2
    bound.hold.release.assert_called_once()


def test_shutdown_drains_before_registration_cleanup(bound: _Fixture) -> None:
    """Shutdown closes admission immediately while retained operations finish later."""
    operation = bound.submit()
    assert not bound.adapter.close()
    assert isinstance(operation.outcome, Cancelled)
    assert bound.backend.registrations[0].calls == 0
    with pytest.raises(RuntimeError, match="not accepting"):
        bound.adapter.submit_fetch(
            _Lease(), bound.backend, bound.watchdog.create_owner(2, "receive", 5)
        )
    bound.finish_proof()
    assert bound.adapter.close()
    assert operation.released
    assert bound.lease.marks[0][0].tolist() == [False, False]


def test_registration_failure_retains_earlier_handles_and_physical_parts(bound: _Fixture) -> None:
    """Identical layout names never collapse distinct allocation registrations."""
    bound.adapter.close()
    bound.events.clear()
    lender = Mock()
    lender.parts = (Part("same layout", 1000, 64, 16, 4), Part("same layout", 2000, 64, 16, 4))
    lender.hold_parts.side_effect = bound.take_hold
    backend = _Backend(bound.events)
    backend.register_fail_at = 1
    adapter = SharedStagingAdapter(
        bound.manager,
        lender,
        bound.profile,
        backend,
        bound.watchdog.create_owner(-2, "receive", 5),
        bound.record_copy,
    )
    with pytest.raises(RuntimeError, match="register failed"):
        adapter.register()
    assert bound.events == ["hold", "register", "register"]
    assert backend.spans == [(1000, 64)]
    assert adapter.close()
    assert bound.events[-2:] == ["deregister", "release hold"]


def test_profile_rejected_before_acquiring_any_hold(bound: _Fixture) -> None:
    """An incompatible profile cannot expose or register manager allocations."""
    lender = Mock()
    with pytest.raises(ValueError, match="cache_dtype"):
        SharedStagingAdapter(
            bound.manager,
            lender,
            replace(bound.profile, cache_dtype="float8"),
            bound.backend,
            bound.registration_retirement,
            bound.record_copy,
        )
    lender.hold_parts.assert_not_called()


def test_same_lease_cannot_be_submitted_twice(bound: _Fixture) -> None:
    """One physical lease cannot be owned by competing operation bindings."""
    bound.submit()
    with pytest.raises(RuntimeError, match="already owned"):
        bound.adapter.submit_fetch(
            bound.lease, bound.backend, bound.watchdog.create_owner(2, "receive", 5)
        )
    assert bound.backend.submissions == 1
    bound.finish_proof()
    assert bound.adapter.close()


def test_retired_operation_and_lease_are_collectible(bound: _Fixture) -> None:
    """The duplicate-deadline guard must not root every previously retired loan."""
    operation = bound.submit()
    lease_ref, operation_ref = weakref.ref(bound.lease), weakref.ref(operation)
    retirement_ref = weakref.ref(operation.retirement)
    bound.backend.attempt.outcome = Delivered(frozenset(_NAMES))
    bound.finish_proof()
    bound.adapter.progress()
    assert operation.released
    del bound.lease, bound.operation, operation
    gc.collect()
    assert operation_ref() is None
    assert lease_ref() is None
    assert retirement_ref() is None


def test_registration_provider_must_match_submission_provider(bound: _Fixture) -> None:
    """A structurally compatible foreign provider cannot use another provider's handles."""
    foreign = _Backend([])
    retirement = bound.watchdog.create_owner(2, "receive", 5)
    with pytest.raises(ValueError, match="own this adapter"):
        bound.adapter.submit_fetch(bound.lease, foreign, retirement)
    assert foreign.submissions == 0
    assert bound.lease.release_count == 0


def test_deadline_reuse_is_rejected_without_overwriting_timeout_binding(bound: _Fixture) -> None:
    """An already bound deadline cannot silently lose the first operation's timeout."""
    first = bound.submit()
    with pytest.raises(RuntimeError, match="own retirement"):
        bound.adapter.submit_fetch(_Lease(), bound.backend, first.retirement)
    bound.clock.return_value = 15.0
    bound.watchdog.progress()
    assert first.outcome == Failed("transfer timeout")
    assert bound.backend.submissions == 1
    bound.finish_proof()
    bound.adapter.progress()
    assert first.released
    assert bound.adapter.close()


def test_registration_and_operation_share_one_containment_controller(bound: _Fixture) -> None:
    """One worker-level containment decision must govern all retained staging roots."""
    foreign = RetirementWatchdog(Mock(), clock=bound.clock)
    with pytest.raises(ValueError, match="retirement watchdog"):
        bound.adapter.submit_fetch(
            bound.lease, bound.backend, foreign.create_owner(3, "receive", 5)
        )
    assert bound.backend.submissions == 0
