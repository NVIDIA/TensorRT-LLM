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
"""Packed attention-DP executor state exchange (TLLM_ADP_PACKED_SYNC).

The disaggregated attention-DP executor loops replace per-iteration object
collectives with two fixed-width int64 tensor all-gathers: the loop-head group
(disagg error vote + KV-timeout vote + RankState) and the post-schedule group
(can_queue + the cuda-graph runner's pre-padding vector). These tests pin the
contract on simulated ranks (one thread per rank, barrier-synchronised fake
collectives):

* the actions taken from the packed vectors equal the legacy functions'
  actions for every combination of error / blocked / timeout votes,
* every consumer follows the stash contract (use if present, else gather), so
  callers outside the loop -- warmup, KV-cache estimation -- gather as before,
* stashes never outlive a pass or an executor instance,
* the gate is config-derived and the knob-off path is byte-identical to today,
* verify mode raises on a mismatch or an unconsumed stash,
* TorchDist's fixed-width int64 gathers ride the single-kernel CPU-tensor path
  when the object-collective groups exist and fall back to the base path
  otherwise.
"""
import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    DisaggTransferCoordinator, NoopDisaggCoordinator)
from tensorrt_llm._torch.distributed.communicator import TorchDist
from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    DefaultADPRouter, RankState)

pytestmark = pytest.mark.cpu_only

# ---------------------------------------------------------------------------
# Simulated TP group
# ---------------------------------------------------------------------------


class _FakeGroup:
    """N simulated ranks; each collective is a barrier over N threads.

    Every rank's dist records ``(op, payload)`` so tests can assert that all
    ranks issued the identical sequence.
    """

    def __init__(self, n):
        self.n = n
        self._barrier = threading.Barrier(n)
        self._slots = [None] * n
        self.ops = [[] for _ in range(n)]
        self.payloads = [[] for _ in range(n)]

    def _exchange(self, rank, op, payload):
        self.ops[rank].append(op)
        self.payloads[rank].append(copy.deepcopy(payload))
        self._slots[rank] = payload
        self._barrier.wait(timeout=30)
        gathered = list(self._slots)
        self._barrier.wait(timeout=30)
        return gathered

    def dist(self, rank):
        d = SimpleNamespace(rank=rank,
                            tp_rank=rank,
                            world_size=self.n,
                            tp_size=self.n,
                            pp_size=1,
                            cp_size=1,
                            has_cp_helix=False)
        d.allgather_ints = lambda values: [
            list(v)
            for v in self._exchange(rank, "allgather_ints", list(values))
        ]
        d.tp_allgather = lambda obj, **kw: self._exchange(
            rank, "tp_allgather", obj)
        d.tp_allgather_int64 = lambda values: np.asarray(
            self._exchange(rank, "tp_allgather_int64", list(values)),
            dtype=np.int64)
        d.allreduce = lambda v, op=None: max(self._exchange(rank, "allreduce", v))
        d.tp_allreduce = lambda v, op=None: max(
            self._exchange(rank, "tp_allreduce", v))
        return d

    def run(self, fns):
        results = [None] * self.n
        errors = [None] * self.n

        def worker(i):
            try:
                results[i] = fns[i]()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[i] = exc
                self._barrier.abort()

        threads = [
            threading.Thread(target=worker, args=(i, )) for i in range(self.n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        for exc in errors:
            if exc is not None and not isinstance(exc,
                                                  threading.BrokenBarrierError):
                raise exc
        for exc in errors:
            if exc is not None:
                raise exc
        return results


@pytest.fixture(autouse=True)
def _knob_env(monkeypatch):
    monkeypatch.setenv("TLLM_ADP_PACKED_SYNC", "1")
    monkeypatch.delenv("TLLM_ADP_PACKED_SYNC_VERIFY", raising=False)
    # The in-flight cancel helper caches its env lookup; pin the gate's view.
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled",
                        lambda: False)


def _req(request_id, *, error=False, prompt_len=4):
    return SimpleNamespace(
        request_id=request_id,
        py_request_id=request_id,
        parent_request_id=request_id,
        is_child=False,
        py_orig_prompt_len=prompt_len,
        py_disaggregated_params=None,
        is_context_only_request=False,
        state=(LlmRequestState.DISAGG_TRANS_ERROR
               if error else LlmRequestState.GENERATION_IN_PROGRESS),
    )


class _Coordinator:
    """Stand-in for the coordinator surface the packed exchange uses."""

    def __init__(self, pending_ctx_failures=(), pending_timed_out=()):
        self._pending_ctx = set(pending_ctx_failures)
        self._timed_out = list(pending_timed_out)
        self.failed_synced = []
        self.handle_errors_synced = Mock()
        self._ex = None

    def bind(self, ex):
        self._ex = ex

    def take_pending_context_failures(self):
        pending, self._pending_ctx = self._pending_ctx, set()
        return pending

    def local_error_vote(self):
        # DisaggTransferCoordinator.local_error_vote over the executor's
        # request list; a request is blocked while the cancel path owns it.
        pending = self.take_pending_context_failures()
        for req in self._ex.active_requests:
            if executor_module.get_unique_rid(req) in pending:
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
        errors = [
            req for req in self._ex.active_requests
            if req.state == LlmRequestState.DISAGG_TRANS_ERROR
        ]
        canceled = getattr(self._ex, "canceled_req_ids", set())
        return {
            "error_ids": [req.py_request_id for req in errors],
            "blocked_ids": [
                req.py_request_id for req in errors
                if req.py_request_id in canceled
            ],
        }

    def take_pending_timed_out(self):
        timed_out, self._timed_out = self._timed_out, []
        return timed_out

    def fail_timed_out_synced(self, requests):
        self.failed_synced.append(list(requests))


def _make_executor(group,
                   rank,
                   *,
                   active_requests,
                   coordinator=None,
                   runner=None,
                   verify=False):
    ex = object.__new__(PyExecutor)
    ex.dist = group.dist(rank)
    ex.enable_attention_dp = True
    ex.active_requests = list(active_requests)
    ex.adp_router = DefaultADPRouter(ex.dist)
    ex.enable_iter_perf_stats = False
    ex._fetch_hint_enabled = False
    ex.iter_counter = 7
    ex._packed_sync_enabled = True
    ex._packed_sync_verify = verify
    ex._prefetched_rank_states = None
    ex._packed_can_queue = None
    coordinator = coordinator or _Coordinator()
    coordinator.bind(ex)
    ex._disagg_coordinator = coordinator
    ex.model_engine = SimpleNamespace(
        make_graph_batch=lambda batch: (batch, frozenset()),
        cuda_graph_runner=runner)
    return ex


def _make_runner(group, rank, *, enabled=True):
    runner = object.__new__(CUDAGraphRunner)
    runner.enabled = enabled
    runner.padding_enabled = False  # _get_padded_batch returns 0 after the gather
    runner.config = SimpleNamespace(enable_attention_dp=True,
                                    mapping=SimpleNamespace(tp_size=group.n),
                                    dist=group.dist(rank))
    runner._adp_graph_batch_hint = None
    runner.is_encoder_decoder = False
    runner.enable_encoder_decoder_mixed_cuda_graph = False
    return runner


def _batch(batch_size, *, can_run_cuda_graph=True):
    return SimpleNamespace(batch_size=batch_size,
                           can_run_cuda_graph=can_run_cuda_graph,
                           num_context_requests=0,
                           num_generation_requests=batch_size,
                           generation_requests=[None] * batch_size,
                           context_requests=[])


# ---------------------------------------------------------------------------
# Loop-head group
# ---------------------------------------------------------------------------


def _run_loop_head(group, executors):

    def step(ex):
        result = ex._adp_loop_head_exchange()
        ex._adp_apply_loop_head_actions(result)
        return result

    return group.run([lambda ex=ex: step(ex) for ex in executors])


def test_loop_head_steady_state_is_one_gather_and_stashes_rank_states():
    group = _FakeGroup(2)
    executors = [
        _make_executor(group, 0, active_requests=[_req(1), _req(2)]),
        _make_executor(group, 1, active_requests=[_req(3, prompt_len=10)]),
    ]
    results = _run_loop_head(group, executors)
    for rank, ex in enumerate(executors):
        assert group.ops[rank] == ["allgather_ints"]
        assert not results[rank].any_error and not results[rank].any_timeout
        ex._disagg_coordinator.handle_errors_synced.assert_not_called()
        assert ex._disagg_coordinator.failed_synced == []
        states = ex._prefetched_rank_states
        assert [s.rank for s in states] == [0, 1]
        assert [s.num_active_requests for s in states] == [2, 1]
        assert [s.num_active_tokens for s in states] == [8, 10]
        assert all(isinstance(s, RankState) for s in states)
    # The packed vector is the RankState followed by the three vote ints.
    width = len(RankState(rank=0).serialize())
    for rank in range(2):
        vec = group.payloads[rank][0]
        assert vec[width:] == [0, 0, 0]


def test_loop_head_error_vote_runs_legacy_handler_on_every_rank():
    group = _FakeGroup(2)
    executors = [
        _make_executor(group, 0, active_requests=[_req(1)]),
        _make_executor(group,
                       1,
                       active_requests=[_req(3), _req(4, error=True)]),
    ]
    results = _run_loop_head(group, executors)
    for rank, ex in enumerate(executors):
        assert results[rank].any_error
        ex._disagg_coordinator.handle_errors_synced.assert_called_once_with()
        # States are stale after the error handler; the fetch gathers again.
        assert ex._prefetched_rank_states is None
    width = len(RankState(rank=0).serialize())
    assert group.payloads[0][0][width:] == [0, 0, 0]
    assert group.payloads[1][0][width:] == [1, 0, 0]


def test_loop_head_pending_context_failure_joins_the_vote():
    group = _FakeGroup(2)
    failing = _req(9)
    executors = [
        _make_executor(group,
                       0,
                       active_requests=[failing],
                       coordinator=_Coordinator(pending_ctx_failures={
                           executor_module.get_unique_rid(failing)
                       })),
        _make_executor(group, 1, active_requests=[_req(3)]),
    ]
    results = _run_loop_head(group, executors)
    assert failing.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert all(r.any_error for r in results)
    for ex in executors:
        ex._disagg_coordinator.handle_errors_synced.assert_called_once_with()


def test_loop_head_blocked_errors_are_counted_separately():
    group = _FakeGroup(2)
    blocked = _req(5, error=True)
    ex0 = _make_executor(group, 0, active_requests=[blocked])
    ex0.canceled_req_ids = {5}
    ex1 = _make_executor(group, 1, active_requests=[])
    _run_loop_head(group, [ex0, ex1])
    width = len(RankState(rank=0).serialize())
    assert group.payloads[0][0][width:] == [1, 1, 0]


def test_loop_head_timeout_vote_fails_drained_buffer_on_every_rank():
    group = _FakeGroup(2)
    timed_out = _req(11)
    coord0 = _Coordinator(pending_timed_out=[timed_out])
    coord1 = _Coordinator()
    executors = [
        _make_executor(group, 0, active_requests=[], coordinator=coord0),
        _make_executor(group, 1, active_requests=[_req(3)], coordinator=coord1),
    ]
    results = _run_loop_head(group, executors)
    assert all(r.any_timeout for r in results)
    # Rank 0 fails its drained request, rank 1 enters the same path with an
    # empty list (rank-symmetric like handle_timeouts_synced).
    assert coord0.failed_synced == [[timed_out]]
    assert coord1.failed_synced == [[]]
    assert coord0._timed_out == []
    for ex in executors:
        ex._disagg_coordinator.handle_errors_synced.assert_not_called()
        assert ex._prefetched_rank_states is not None


def test_loop_head_verify_mode_issues_legacy_collectives_and_agrees():
    group = _FakeGroup(2)
    executors = [
        _make_executor(group, 0, active_requests=[_req(1)], verify=True),
        _make_executor(group, 1, active_requests=[_req(2, error=True)],
                       verify=True),
    ]
    _run_loop_head(group, executors)
    for rank in range(2):
        assert group.ops[rank] == [
            "allgather_ints", "tp_allgather", "tp_allgather_int64",
            "tp_allgather"
        ]


# ---------------------------------------------------------------------------
# Post-schedule group
# ---------------------------------------------------------------------------


def test_post_schedule_exchange_stashes_can_queue_and_padding_hint():
    group = _FakeGroup(2)
    runners = [_make_runner(group, r) for r in range(2)]
    executors = [
        _make_executor(group, r, active_requests=[], runner=runners[r])
        for r in range(2)
    ]
    batches = [_batch(3), _batch(0, can_run_cuda_graph=False)]

    def step(rank):
        ex = executors[rank]
        ex._adp_post_schedule_exchange(batches[rank])
        can_queue = ex._can_queue(batches[rank])
        # The runner pops the hint instead of gathering.
        runners[rank]._gather_adp_graph_batch_info = Mock(
            side_effect=AssertionError("must not gather with a hint"))
        padding = runners[rank]._get_padded_batch(batches[rank], None, 0)
        return can_queue, padding

    results = group.run([lambda r=r: step(r) for r in range(2)])
    assert results[0] == ((False, True), 0)
    assert results[1] == ((False, False), 0)
    for rank in range(2):
        # One packed gather; _can_queue and the runner issued nothing.
        assert group.ops[rank] == ["allgather_ints"]
        assert group.payloads[rank][0] == [
            batches[rank].batch_size,
            int(batches[rank].can_run_cuda_graph), batches[rank].batch_size
        ]
        assert executors[rank]._packed_can_queue is None
        assert runners[rank]._adp_graph_batch_hint is None


def test_padding_hint_rows_match_the_runner_gather_layout():
    group = _FakeGroup(2)
    runners = [_make_runner(group, r) for r in range(2)]
    executors = [
        _make_executor(group, r, active_requests=[], runner=runners[r])
        for r in range(2)
    ]
    batches = [_batch(2), _batch(5)]
    group.run([
        lambda r=r: executors[r]._adp_post_schedule_exchange(batches[r])
        for r in range(2)
    ])
    for rank in range(2):
        assert runners[rank]._adp_graph_batch_hint == [
            (True, 2, SampleType.FULL.value), (True, 5, SampleType.FULL.value)
        ]


def test_runner_without_hint_gathers_as_before():
    group = _FakeGroup(2)
    runner = _make_runner(group, 0)
    runner._gather_adp_graph_batch_info = Mock(return_value=[(True, 4, 0)])
    assert runner._get_padded_batch(_batch(4), None, 0) == 0
    runner._gather_adp_graph_batch_info.assert_called_once()
    assert runner._adp_graph_batch_hint is None


def test_can_queue_without_stash_gathers_as_before():
    ex = object.__new__(PyExecutor)
    ex.enable_attention_dp = True
    ex._packed_can_queue = None
    ex.dist = SimpleNamespace(tp_allgather_int64=Mock(
        return_value=np.array([[2], [0]])))
    assert PyExecutor._can_queue(ex, SimpleNamespace(batch_size=2)) == (False,
                                                                        True)
    ex.dist.tp_allgather_int64.assert_called_once_with([2])


# ---------------------------------------------------------------------------
# Stash lifecycle, verify mode, gate
# ---------------------------------------------------------------------------


def test_end_of_pass_clear_drops_every_stash_and_verify_flags_leftovers():
    group = _FakeGroup(1)
    runner = _make_runner(group, 0)
    ex = _make_executor(group, 0, active_requests=[], runner=runner)
    ex._prefetched_rank_states = [RankState(rank=0)]
    ex._packed_can_queue = (True, True)
    runner._adp_graph_batch_hint = [(True, 1, 0)]
    ex._adp_clear_packed_sync_state()
    assert ex._prefetched_rank_states is None
    assert ex._packed_can_queue is None
    assert runner._adp_graph_batch_hint is None

    ex._packed_sync_verify = True
    ex._packed_can_queue = (True, True)
    with pytest.raises(RuntimeError, match="can_queue stash not consumed"):
        ex._adp_clear_packed_sync_state()
    ex._packed_can_queue = (True, True)
    ex._adp_clear_packed_sync_state(check=False)  # cleanup path never raises
    assert ex._packed_can_queue is None

    # Pre-warmup reset drops a leftover hint on the shared runner.
    runner._adp_graph_batch_hint = [(True, 1, 0)]
    ex._reset_adp_packed_sync_for_warmup()
    assert runner._adp_graph_batch_hint is None


def test_packed_sync_check_raises_on_mismatch():
    ex = object.__new__(PyExecutor)
    ex.dist = SimpleNamespace(rank=1)
    ex.iter_counter = 3
    ex._packed_sync_check("f", [1], [1])
    with pytest.raises(RuntimeError, match="f mismatch on rank 1"):
        ex._packed_sync_check("f", [1], [2])


def _gate_executor(monkeypatch, **overrides):
    ex = object.__new__(PyExecutor)
    ex.dist = SimpleNamespace(rank=0,
                              world_size=2,
                              tp_size=2,
                              pp_size=1,
                              cp_size=1)
    ex.enable_attention_dp = True
    ex.kv_cache_transceiver = object()
    ex.kv_connector_manager = None
    ex.is_benchmark_disagg = False
    ex.drafter = None
    ex.enable_early_first_token_response = False
    ex.model_engine = SimpleNamespace(is_spec_decode=False,
                                      make_graph_batch=lambda b: (b, frozenset()))
    ex.adp_router = DefaultADPRouter(ex.dist)
    for key, value in overrides.items():
        setattr(ex, key, value)
    return ex


def test_gate_is_config_derived(monkeypatch):
    assert _gate_executor(monkeypatch)._resolve_packed_sync_enabled() is True
    monkeypatch.setenv("TLLM_ADP_PACKED_SYNC", "0")
    assert _gate_executor(monkeypatch)._resolve_packed_sync_enabled() is False
    monkeypatch.setenv("TLLM_ADP_PACKED_SYNC", "1")
    for override in (
            dict(enable_attention_dp=False),
            dict(kv_cache_transceiver=None),
            dict(kv_connector_manager=object()),
            dict(is_benchmark_disagg=True),
            dict(drafter=object()),
            dict(enable_early_first_token_response=True),
            dict(model_engine=SimpleNamespace(is_spec_decode=True,
                                              make_graph_batch=lambda b: b)),
            dict(model_engine=SimpleNamespace(is_spec_decode=False)),
            dict(dist=SimpleNamespace(
                rank=0, world_size=4, tp_size=2, pp_size=2, cp_size=1)),
            dict(dist=SimpleNamespace(
                rank=0, world_size=1, tp_size=1, pp_size=1, cp_size=1)),
    ):
        assert _gate_executor(
            monkeypatch, **override)._resolve_packed_sync_enabled() is False
    monkeypatch.setattr(executor_module, "is_disagg_inflight_cancel_enabled",
                        lambda: True)
    assert _gate_executor(monkeypatch)._resolve_packed_sync_enabled() is False


# ---------------------------------------------------------------------------
# Coordinator accessors
# ---------------------------------------------------------------------------


def test_coordinator_timeout_accessors():
    coord = object.__new__(DisaggTransferCoordinator)
    coord._pending_timed_out_requests = ["a", "b"]
    coord._effects = SimpleNamespace(fail_requests=Mock())
    assert coord.take_pending_timed_out() == ["a", "b"]
    assert coord._pending_timed_out_requests == []
    coord.fail_timed_out_synced([])
    coord.fail_timed_out_synced(["a"])
    assert coord._effects.fail_requests.call_count == 2
    coord._effects.fail_requests.assert_called_with(
        "Request timed out (KV transfer)", ["a"], charge_budget=False)

    noop = NoopDisaggCoordinator()
    assert noop.take_pending_timed_out() == []
    assert noop.fail_timed_out_synced(["x"]) is None


def test_coordinator_local_error_vote_marks_pending_failures_and_counts_blocked():
    ok, failing, blocked = _req(1), _req(2), _req(3, error=True)
    coord = object.__new__(DisaggTransferCoordinator)
    coord._pending_ctx_transfer_failures = {executor_module.get_unique_rid(failing)}
    coord._registry = SimpleNamespace(
        active_requests=lambda: [ok, failing, blocked],
        canceled_request_ids=lambda: {3})
    coord._transfers = SimpleNamespace(requests_in_transfer=lambda: set())
    vote = coord.local_error_vote()
    # The pending context failure joined the vote and the buffer is drained;
    # the canceled request is voted but reported as blocked.
    assert failing.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert ok.state == LlmRequestState.GENERATION_IN_PROGRESS
    assert vote == {"error_ids": [2, 3], "blocked_ids": [3]}
    assert coord.take_pending_context_failures() == set()

    noop = NoopDisaggCoordinator()
    assert noop.local_error_vote() == {"error_ids": [], "blocked_ids": []}


# ---------------------------------------------------------------------------
# TorchDist fixed-width int64 gather routing
# ---------------------------------------------------------------------------


def test_torchdist_int64_gather_uses_cpu_tensor_path_when_available():
    dist = object.__new__(TorchDist)
    pg = object()
    dist._cuda_obj_pg = lambda dim: pg if dim == "tp" else None
    sent = []

    def fake_cpu_tensor_gather(tensor, group):
        assert group is pg and tensor.dtype == torch.int64
        sent.append(tensor.tolist())
        return [tensor.clone(), tensor.clone() + 10]

    dist._cuda_allgather_cpu_tensor = fake_cpu_tensor_gather
    dist.tp_allgather = Mock(side_effect=AssertionError("object path used"))
    rows = dist.tp_allgather_int64([3, 4])
    assert rows.dtype == np.int64 and rows.tolist() == [[3, 4], [13, 14]]
    assert sent == [[3, 4]]


def test_torchdist_int64_gather_falls_back_without_vote_group():
    dist = object.__new__(TorchDist)
    dist._cuda_obj_pg = lambda dim: None
    dist.tp_allgather = Mock(return_value=[[3, 4], [5, 6]])
    rows = dist.tp_allgather_int64([3, 4])
    assert rows.tolist() == [[3, 4], [5, 6]]
    dist.tp_allgather.assert_called_once_with([3, 4], small_payload=True)
