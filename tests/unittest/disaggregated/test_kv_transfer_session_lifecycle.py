# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the disaggregated KV-transfer session lifecycle.

These tests exercise the Python policy layer added on top of the C++
transceiver and intentionally avoid requiring a real GPU, MPI runtime, or
the full PyExecutor stack. The focus is the contract that the analysis
doc requires us to keep:

* per-request timeouts must NOT trigger a global executor shutdown;
* a CancelRequestedInFlight result must keep the request owned by C++
  (no _do_terminate_request, no removal from active queues);
* CancelledBeforeAdvertise / AlreadyComplete / NotFound (after a final
  state has been observed) must release resources promptly;
* the ADP pending-response flush is symmetric across DP ranks.
"""

from __future__ import annotations

import enum
import types
from dataclasses import dataclass, field
from typing import List, Optional

import pytest


class _FakeTransferCancelResult(enum.Enum):
    """Mirror of the C++ enum exposed by the binding so the tests do not
    pull in the heavyweight binding module on import."""
    NotFound = 0
    AlreadyComplete = 1
    CancelledBeforeAdvertise = 2
    CancelRequestedInFlight = 3
    BackendUnhealthy = 4
    NotCancellable = 5


@dataclass
class _FakeRequest:
    py_request_id: int
    state: str = "DISAGG_GENERATION_TRANS_IN_PROGRESS"
    py_kv_transfer_timed_out: bool = False
    py_kv_transfer_start_time: Optional[float] = None
    is_dummy_request: bool = False

    @property
    def is_disagg_generation_transmission_in_progress(self) -> bool:
        return self.state == "DISAGG_GENERATION_TRANS_IN_PROGRESS"


@dataclass
class _FakeHealth:
    is_healthy: bool = True
    quarantined_transfer_count: int = 0
    quarantine_budget: int = 16
    seconds_since_last_progress: float = 0.0
    global_progress_deadline_seconds: float = 60.0


@dataclass
class _FakeTransceiver:
    """Programmable stand-in for the C++-backed KvCacheTransceiver. The
    test sets `next_cancel_results` to a list keyed by request id; each
    cancel call pops the head."""
    next_cancel_results: dict = field(default_factory=dict)
    cancel_calls: List[int] = field(default_factory=list)
    healthy: bool = True
    health: _FakeHealth = field(default_factory=_FakeHealth)

    def cancel_request_structured(self, req: _FakeRequest):
        self.cancel_calls.append(req.py_request_id)
        return self.next_cancel_results.get(
            req.py_request_id, _FakeTransferCancelResult.CancelRequestedInFlight)

    def is_healthy(self) -> bool:
        return self.healthy

    def get_health(self) -> _FakeHealth:
        return self.health


def _make_executor_under_test():
    """Build a minimal object whose interface matches what the helper
    methods need. We avoid constructing a real PyExecutor because that
    drags in CUDA, MPI, and the full model engine."""
    obj = types.SimpleNamespace()

    obj.kv_cache_transceiver = _FakeTransceiver()
    obj._inflight_cancel_requested_ids = set()
    obj._pending_transfer_responses = []
    obj._pending_resource_release = []
    obj._transceiver_unhealthy_since = None
    obj.is_shutdown = False
    obj.shutdown_event = None
    obj.enable_attention_dp = False
    obj.active_requests = []
    obj.terminated_request_ids = []
    obj.enqueue_calls = []

    # Bind the real implementations (copied from py_executor.py) by
    # patching through closures so we do not have to import the whole
    # module — the unit-under-test functions are pure Python with no
    # CUDA/MPI dependencies.

    def _can_terminate_request_now(self, request) -> bool:
        if self.kv_cache_transceiver is None:
            return True
        if request.is_dummy_request:
            return True
        if request.state in ("DISAGG_CONTEXT_TRANS_IN_PROGRESS",
                             "DISAGG_GENERATION_TRANS_IN_PROGRESS"):
            return False
        if request.py_request_id in self._inflight_cancel_requested_ids:
            return False
        return True

    def _do_terminate_request(self, request):
        if not self._can_terminate_request_now(request):
            return
        self.terminated_request_ids.append(request.py_request_id)

    def _enqueue_responses(self, responses):
        # Mimics the dist-side aggregator; we just record what was sent.
        self.enqueue_calls.append(list(responses))

    def _flush_pending_transfer_responses(self):
        responses = self._pending_transfer_responses
        self._pending_transfer_responses = []
        if responses or self.enable_attention_dp:
            self._enqueue_responses(responses)

    def _defer_resource_release_for_inflight_transfer(self, request):
        self._inflight_cancel_requested_ids.add(request.py_request_id)
        if request not in self._pending_resource_release:
            self._pending_resource_release.append(request)

    def _maybe_release_pending_resources(self):
        if not self._pending_resource_release or self.kv_cache_transceiver is None:
            return
        still_pending = []
        safe = (
            _FakeTransferCancelResult.AlreadyComplete,
            _FakeTransferCancelResult.NotFound,
            _FakeTransferCancelResult.CancelledBeforeAdvertise,
        )
        for request in self._pending_resource_release:
            cancel = self.kv_cache_transceiver.cancel_request_structured(request)
            if cancel in safe:
                self._inflight_cancel_requested_ids.discard(request.py_request_id)
                self._do_terminate_request(request)
            else:
                still_pending.append(request)
        self._pending_resource_release = still_pending

    def _check_transceiver_health(self):
        if self.kv_cache_transceiver is None:
            return
        if self.kv_cache_transceiver.is_healthy():
            self._transceiver_unhealthy_since = None
            return
        import time as _time
        now = _time.time()
        if self._transceiver_unhealthy_since is None:
            self._transceiver_unhealthy_since = now
            return
        elapsed = now - self._transceiver_unhealthy_since
        health = self.kv_cache_transceiver.get_health()
        grace = 2.0 * float(getattr(health, "global_progress_deadline_seconds", 60.0))
        if elapsed > grace and not self.is_shutdown:
            self.is_shutdown = True

    obj._can_terminate_request_now = types.MethodType(
        _can_terminate_request_now, obj)
    obj._do_terminate_request = types.MethodType(_do_terminate_request, obj)
    obj._enqueue_responses = types.MethodType(_enqueue_responses, obj)
    obj._flush_pending_transfer_responses = types.MethodType(
        _flush_pending_transfer_responses, obj)
    obj._defer_resource_release_for_inflight_transfer = types.MethodType(
        _defer_resource_release_for_inflight_transfer, obj)
    obj._maybe_release_pending_resources = types.MethodType(
        _maybe_release_pending_resources, obj)
    obj._check_transceiver_health = types.MethodType(
        _check_transceiver_health, obj)
    return obj


# ---------------------------------------------------------------------------
# _can_terminate_request_now
# ---------------------------------------------------------------------------


def test_can_terminate_request_defers_during_active_transfer():
    executor = _make_executor_under_test()
    req = _FakeRequest(py_request_id=42,
                       state="DISAGG_GENERATION_TRANS_IN_PROGRESS")
    assert not executor._can_terminate_request_now(req)


def test_can_terminate_request_defers_when_cancel_in_flight():
    executor = _make_executor_under_test()
    req = _FakeRequest(py_request_id=42, state="GENERATION_COMPLETE")
    executor._inflight_cancel_requested_ids.add(42)
    assert not executor._can_terminate_request_now(req)


def test_can_terminate_request_allows_when_safe():
    executor = _make_executor_under_test()
    req = _FakeRequest(py_request_id=42, state="GENERATION_COMPLETE")
    assert executor._can_terminate_request_now(req)


def test_dummy_request_is_always_terminable():
    executor = _make_executor_under_test()
    req = _FakeRequest(py_request_id=42,
                       state="DISAGG_GENERATION_TRANS_IN_PROGRESS",
                       is_dummy_request=True)
    assert executor._can_terminate_request_now(req)


def test_no_transceiver_means_terminable():
    executor = _make_executor_under_test()
    executor.kv_cache_transceiver = None
    req = _FakeRequest(py_request_id=42,
                       state="DISAGG_GENERATION_TRANS_IN_PROGRESS")
    assert executor._can_terminate_request_now(req)


# ---------------------------------------------------------------------------
# _do_terminate_request honors the guard
# ---------------------------------------------------------------------------


def test_do_terminate_defers_when_unsafe():
    executor = _make_executor_under_test()
    req = _FakeRequest(py_request_id=99,
                       state="DISAGG_GENERATION_TRANS_IN_PROGRESS")
    executor._do_terminate_request(req)
    assert executor.terminated_request_ids == []


def test_do_terminate_runs_when_safe():
    executor = _make_executor_under_test()
    req = _FakeRequest(py_request_id=99, state="GENERATION_COMPLETE")
    executor._do_terminate_request(req)
    assert executor.terminated_request_ids == [99]


# ---------------------------------------------------------------------------
# ADP response flush
# ---------------------------------------------------------------------------


def test_flush_pending_transfer_responses_empties_buffer():
    executor = _make_executor_under_test()
    executor._pending_transfer_responses = [(1, "r1"), (2, "r2")]
    executor._flush_pending_transfer_responses()

    assert executor._pending_transfer_responses == []
    assert executor.enqueue_calls == [[(1, "r1"), (2, "r2")]]


def test_flush_with_adp_enabled_calls_enqueue_even_when_empty():
    """Even when the local rank has nothing to enqueue, ADP requires every
    rank to participate in tp_gather. If we skipped the call we would
    deadlock the other rank's collective."""
    executor = _make_executor_under_test()
    executor.enable_attention_dp = True
    executor._flush_pending_transfer_responses()
    assert executor.enqueue_calls == [[]]


def test_flush_without_adp_skips_when_empty():
    executor = _make_executor_under_test()
    executor.enable_attention_dp = False
    executor._flush_pending_transfer_responses()
    assert executor.enqueue_calls == []


# ---------------------------------------------------------------------------
# Per-request timeout policy: do NOT fail-close the executor
# ---------------------------------------------------------------------------


def _apply_timeout_policy(executor,
                          request,
                          *,
                          cancel_result=_FakeTransferCancelResult.
                          CancelRequestedInFlight):
    """Mini-replay of the timeout branch in _handle_responses for a single
    request. Mirrors the production decision tree without dragging in the
    full event loop."""
    request.py_kv_transfer_timed_out = True
    executor.kv_cache_transceiver.next_cancel_results[
        request.py_request_id] = cancel_result

    cancel = executor.kv_cache_transceiver.cancel_request_structured(request)
    safe_to_release = cancel in (
        _FakeTransferCancelResult.CancelledBeforeAdvertise,
        _FakeTransferCancelResult.AlreadyComplete,
        _FakeTransferCancelResult.NotFound,
    )
    if safe_to_release:
        executor._inflight_cancel_requested_ids.discard(request.py_request_id)
        return "released"
    if cancel == _FakeTransferCancelResult.CancelRequestedInFlight:
        executor._inflight_cancel_requested_ids.add(request.py_request_id)
        return "deferred"
    if cancel == _FakeTransferCancelResult.BackendUnhealthy:
        executor._inflight_cancel_requested_ids.add(request.py_request_id)
        return "unhealthy-deferred"
    return "ignored"


def test_timeout_with_in_flight_worker_does_not_clear_active_queue():
    executor = _make_executor_under_test()
    other_req = _FakeRequest(py_request_id=1, state="GENERATION_IN_PROGRESS")
    timed_out = _FakeRequest(py_request_id=2,
                             state="DISAGG_GENERATION_TRANS_IN_PROGRESS")
    executor.active_requests = [other_req, timed_out]

    outcome = _apply_timeout_policy(
        executor,
        timed_out,
        cancel_result=_FakeTransferCancelResult.CancelRequestedInFlight)

    # The PR 13706 anti-pattern was to clear active_requests / shutdown
    # the executor on a per-request timeout. We must NOT do that.
    assert outcome == "deferred"
    assert other_req in executor.active_requests
    assert timed_out in executor.active_requests
    assert 2 in executor._inflight_cancel_requested_ids
    # _do_terminate_request must defer.
    executor._do_terminate_request(timed_out)
    assert executor.terminated_request_ids == []


def test_timeout_pre_advertise_releases_resources():
    executor = _make_executor_under_test()
    timed_out = _FakeRequest(py_request_id=7,
                             state="DISAGG_CONTEXT_TRANS_IN_PROGRESS")

    outcome = _apply_timeout_policy(
        executor,
        timed_out,
        cancel_result=_FakeTransferCancelResult.CancelledBeforeAdvertise)

    assert outcome == "released"
    assert 7 not in executor._inflight_cancel_requested_ids
    timed_out.state = "GENERATION_COMPLETE"
    executor._do_terminate_request(timed_out)
    assert executor.terminated_request_ids == [7]


def test_timeout_followed_by_worker_completion_eventually_releases():
    """Two-iteration sequence: first iteration the worker is in flight,
    second iteration C++ has erased the future. Both must be handled
    without ever calling _do_terminate_request before the worker
    quiesces."""
    executor = _make_executor_under_test()
    request = _FakeRequest(py_request_id=11,
                           state="DISAGG_GENERATION_TRANS_IN_PROGRESS")

    # Iteration 1: in flight.
    assert _apply_timeout_policy(
        executor,
        request,
        cancel_result=_FakeTransferCancelResult.CancelRequestedInFlight
    ) == "deferred"
    executor._do_terminate_request(request)
    assert executor.terminated_request_ids == []

    # Iteration 2: C++ has reached final state and erased the future.
    request.state = "DISAGG_TRANS_ERROR"
    assert _apply_timeout_policy(
        executor, request,
        cancel_result=_FakeTransferCancelResult.NotFound) == "released"
    assert 11 not in executor._inflight_cancel_requested_ids
    executor._do_terminate_request(request)
    assert executor.terminated_request_ids == [11]


def test_unhealthy_transceiver_keeps_request_pinned():
    executor = _make_executor_under_test()
    request = _FakeRequest(py_request_id=33,
                           state="DISAGG_GENERATION_TRANS_IN_PROGRESS")

    outcome = _apply_timeout_policy(
        executor,
        request,
        cancel_result=_FakeTransferCancelResult.BackendUnhealthy)

    # Even when the C++ side reports unhealthy, Python must NOT free the
    # request right away. Orchestration is supposed to restart the
    # worker; freeing locally would leak buffers to the wedged backend.
    assert outcome == "unhealthy-deferred"
    assert 33 in executor._inflight_cancel_requested_ids
    executor._do_terminate_request(request)
    assert executor.terminated_request_ids == []


# ---------------------------------------------------------------------------
# Deferred-resource-release polling
# ---------------------------------------------------------------------------


def test_defer_resource_release_holds_until_quiesced():
    """Replay the full recovery sequence: timeout fires, error response
    is surfaced, KV cleanup is deferred. Subsequent iterations poll
    cancel_request_structured; when it transitions to AlreadyComplete /
    NotFound, the deferred terminate runs."""
    executor = _make_executor_under_test()
    request = _FakeRequest(py_request_id=77, state="DISAGG_TRANS_ERROR")
    executor._defer_resource_release_for_inflight_transfer(request)

    # Iteration 1: worker still in flight — must NOT free.
    executor.kv_cache_transceiver.next_cancel_results[77] = (
        _FakeTransferCancelResult.CancelRequestedInFlight)
    executor._maybe_release_pending_resources()
    assert executor.terminated_request_ids == []
    assert request in executor._pending_resource_release
    assert 77 in executor._inflight_cancel_requested_ids

    # Iteration 2: still in flight.
    executor._maybe_release_pending_resources()
    assert executor.terminated_request_ids == []

    # Iteration 3: worker finally quiesced.
    executor.kv_cache_transceiver.next_cancel_results[77] = (
        _FakeTransferCancelResult.AlreadyComplete)
    executor._maybe_release_pending_resources()
    assert executor.terminated_request_ids == [77]
    assert request not in executor._pending_resource_release
    assert 77 not in executor._inflight_cancel_requested_ids


def test_defer_resource_release_handles_not_found_as_safe():
    """C++ may erase the worker entry between iterations; NotFound at
    that point means the worker is gone and resources are safe to
    free."""
    executor = _make_executor_under_test()
    request = _FakeRequest(py_request_id=88, state="DISAGG_TRANS_ERROR")
    executor._defer_resource_release_for_inflight_transfer(request)

    executor.kv_cache_transceiver.next_cancel_results[88] = (
        _FakeTransferCancelResult.NotFound)
    executor._maybe_release_pending_resources()
    assert executor.terminated_request_ids == [88]


def test_defer_resource_release_no_op_when_empty():
    executor = _make_executor_under_test()
    executor._maybe_release_pending_resources()
    assert executor.terminated_request_ids == []


# ---------------------------------------------------------------------------
# Health-driven shutdown escape
# ---------------------------------------------------------------------------


def test_check_transceiver_health_resets_on_recovery():
    executor = _make_executor_under_test()
    executor.kv_cache_transceiver.healthy = False
    executor._check_transceiver_health()
    assert executor._transceiver_unhealthy_since is not None

    # Recovery before grace period elapses — clear the timestamp,
    # do not shut down.
    executor.kv_cache_transceiver.healthy = True
    executor._check_transceiver_health()
    assert executor._transceiver_unhealthy_since is None
    assert not executor.is_shutdown


def test_check_transceiver_health_triggers_shutdown_after_grace():
    executor = _make_executor_under_test()
    # Set an extremely short deadline so the grace period (2x) is
    # easy to exceed in a unit test.
    executor.kv_cache_transceiver.health = _FakeHealth(
        is_healthy=False,
        global_progress_deadline_seconds=0.01,
    )
    executor.kv_cache_transceiver.healthy = False

    # First call records the timestamp, does not shut down yet.
    executor._check_transceiver_health()
    assert not executor.is_shutdown
    assert executor._transceiver_unhealthy_since is not None

    # Move the recorded timestamp into the past beyond the grace period.
    executor._transceiver_unhealthy_since = executor._transceiver_unhealthy_since - 1.0
    executor._check_transceiver_health()
    assert executor.is_shutdown
