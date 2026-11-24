# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overlap loop keeps ``py_batch_idx`` only on requests of the in-flight batch.

With the overlap scheduler a step is packed before the previous step is
applied on the host. The packer takes a request's last tokens and accepted
count from the previous step's device tensors when the request has a
``py_batch_idx`` (the in-flight branch of ``engine/runners/decoder/inputs.py``)
and builds the row from the host otherwise. A request left out of a step keeps
the index of an earlier step. Once that step is applied, the in-flight branch
adds its tokens a second time: positions and KV lengths run ahead and the
output is wrong, with no error.

The traces drive the real ``_executor_loop_overlap`` and its two drains over
stand-ins for everything else. A fake forward makes the packer's branch choice
and checks every generation row against the request's true length; the real V2
scheduler leaves a request out through its STOP and PEFT exits. Without
the clear every trace must fail.
"""

import contextlib
import dataclasses
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import (
    BatchState,
    PyExecutor,
    _clear_batch_idx_outside,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

pytestmark = pytest.mark.cpu_only

MAX_DRAFT_LEN = 3
PROMPT_LEN = 10


def _new_tokens(step: int, req) -> int:
    """Tokens a generation step emits for a request: the target token plus a
    step-dependent number of accepted drafts, so a doubled step always shows."""
    return 1 + (7 * step + 3 * req.py_request_id) % (MAX_DRAFT_LEN + 1)


def _request(rid: int, lora_task_id: Optional[int] = None) -> Mock:
    """A generation request, with what the V2 scheduler and the packer read."""
    req = Mock()
    req.request_id = req.py_request_id = rid
    req.py_seq_slot = 10 + rid
    req.py_batch_idx = None
    req.is_dummy = False
    req.state_value = LlmRequestState.GENERATION_IN_PROGRESS.value
    req.py_csa2_remote_tail_mode = None
    req.get_beam_width_by_iter.return_value = 1
    req.py_draft_tokens = [0] * MAX_DRAFT_LEN
    req.lora_task_id = lora_task_id
    req.is_context_init_state = False
    req.is_generation_in_progress_state = True
    req.is_generation_to_complete_state = False
    req.is_first_context_chunk = False
    req.is_last_context_chunk = False
    req.py_encoder_output_ready_event = None
    req.py_multimodal_data = None
    return req


def _batch(gen=(), ctx=()) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = list(ctx)
    batch.generation_requests = list(gen)
    return batch


def test_only_requests_of_the_in_flight_batch_keep_their_index():
    a, b, c, x, d = (_request(rid) for rid in range(5))
    for req in (a, b, c, x, d):
        req.py_batch_idx = req.py_seq_slot
    # c ran its last context chunk in the in-flight step and decodes now.
    previous = BatchState(scheduled_requests=_batch(gen=[a, b], ctx=[c]), sample_state=None)
    current = _batch(gen=[a, c, x], ctx=[d])

    _clear_batch_idx_outside(current, previous)

    assert (a.py_batch_idx, c.py_batch_idx) == (a.py_seq_slot, c.py_seq_slot)
    assert x.py_batch_idx is None and d.py_batch_idx is None
    # A request outside the current batch is left alone.
    assert b.py_batch_idx == b.py_seq_slot


def test_without_an_in_flight_batch_no_request_keeps_its_index():
    a, x = _request(0), _request(1)
    a.py_batch_idx, x.py_batch_idx = a.py_seq_slot, x.py_seq_slot

    _clear_batch_idx_outside(_batch(gen=[a, x]), None)

    assert a.py_batch_idx is None and x.py_batch_idx is None


def test_membership_is_by_request_object_not_by_id():
    # Attention-DP dummies share request id 0; an id match proves nothing.
    in_flight, other = _request(0), _request(0)
    in_flight.py_batch_idx = other.py_batch_idx = in_flight.py_seq_slot
    previous = BatchState(scheduled_requests=_batch(gen=[in_flight]), sample_state=None)

    _clear_batch_idx_outside(_batch(gen=[in_flight, other]), previous)

    assert in_flight.py_batch_idx == in_flight.py_seq_slot
    assert other.py_batch_idx is None


@dataclasses.dataclass
class _Step:
    """One loop iteration. ``gen`` / ``ctx`` script the scheduler's output (request
    ids in scheduler order); with the real scheduler ``leave_out`` names the
    request its exit leaves out instead."""

    gen: tuple = ()
    ctx: tuple = ()
    can_queue: tuple = (True, True)  # (whole group, this rank)
    fail_forward: bool = False
    control_drain: bool = False
    rebalance_drain: bool = False
    leave_out: Optional[int] = None


class _DeviceStore(dict):
    """Persistent per-slot store of the tokens the last sampled step emitted."""

    def __bool__(self):
        return True


_NOOP_METHODS = (
    "_add_kv_cache_events",
    "_commit_kv_cache_stats",
    "_enqueue_responses",
    "_finalize_adp_dummy_allocation",
    "_flush_iter_stats_synced",
    "_flush_pending_transfer_responses",
    "_handle_dynamic_draft_len",
    "_handle_guided_decoder_errors",
    "_kv_connector_terminate_requests",
    "_maybe_prefetch_next_iter_mm_encoders",
    "_maybe_record_hang_diagnostic_phase",
    "_pause_recompute_paused_requests",
    "_prepare_disagg_gen_transmission_complete",
    "_process_previous_batch",
    "_release_unused_connector_reservations",
    "_revert_gen_alloc",
    "_send_kv_async",
    "_terminate_recompute_paused_requests",
    "_update_generation_requests_that_will_complete_next_iteration",
    "_update_request_states",
    "_update_v2_context_resources",
    "_wait_for_model_engine_input_copy",
)


class _Executor(PyExecutor):
    """``PyExecutor``'s own overlap loop and drains over scripted collaborators.

    ``__init__`` sets only what the loop reads. The fake forward stands in for
    the packer and records every row it would pack wrongly in ``errors``.
    """

    # Plain attributes in place of PyExecutor's properties.
    _kv_connector_moves_kv = False
    disagg = None

    def __init__(self, requests, steps, scheduler=None):
        self.requests = {req.py_request_id: req for req in requests}
        self.steps = list(steps)
        self.scheduler = scheduler
        self.step_no = 0
        self.step = None
        self.failed = set()
        self.host_len = {rid: PROMPT_LEN for rid in self.requests}
        self.true_len = dict(self.host_len)
        self.store = _DeviceStore()
        self.errors = []
        self.forward_orders = []
        # Read by the loop and the drains.
        self.device_id = 0
        self.dist = SimpleNamespace(pp_size=1, tp_size=1, cp_size=1)
        self.disagg = SimpleNamespace(
            handle_errors_synced=lambda: None,
            handle_timeouts_synced=lambda: None,
            check_transfer_timeouts=lambda **kwargs: None,
            pace_idle=lambda: None,
        )
        self.hang_detector = _NullHangDetector()
        self.perf_manager = SimpleNamespace(
            enabled=False,
            create_timing_events=lambda: (None, None, None),
            record_perf_events=lambda *args: contextlib.nullcontext(
                SimpleNamespace(start_time=0.0, end_time=0.0)
            ),
            release_forward_timing_events=lambda *args: None,
            compute_batch_gpu_times=lambda *args: None,
        )
        self.resource_manager = SimpleNamespace(prepare_resources=lambda batch: None)
        self.sampler = object()
        self.execution_stream = None
        self.previous_batch = None
        self.has_previous_draft_tokens = False
        self.iter_counter = 0
        self.is_benchmark_disagg = False
        self.enable_iter_perf_stats = False
        self.enable_attention_dp = True
        self.enable_early_first_token_response = False
        self.enable_kv_cache_events = False
        self.disable_overlap_scheduler = False
        self._resource_governor_enabled = False
        self._is_kv_manager_v2 = True
        self._mm_encoder_item_scheduling_enabled = False
        self.kv_cache_transceiver = None
        self.kv_connector_manager = None
        self.guided_decoder = None
        self.speculation_gate = None
        self.drafter = None
        self.use_spec_decode = True

    def _prepare_and_schedule_batch(self):
        if not self.steps:
            return None, None
        self.step = self.steps.pop(0)
        self.step_no += 1
        if self.step.control_drain:
            # _prepare_and_schedule_batch -> _handle_control_request does this.
            self._retire_inflight_batch_for_control()
        return self._schedule(), None

    def _schedule(self):
        if self.scheduler is None:
            return _batch(
                gen=[self.requests[rid] for rid in self.step.gen],
                ctx=[self.requests[rid] for rid in self.step.ctx],
            )
        live = [req for rid, req in self.requests.items() if rid not in self.failed]
        out = self.scheduler.schedule(live, self.step.leave_out)
        return _batch(gen=out.generation_requests, ctx=out.context_requests)

    def _can_queue(self, scheduled_batch):
        return self.step.can_queue

    def _kv_connector_start_batch(self, scheduled_batch):
        return True, True

    def _can_pause_for_rebalance(self):
        # Asked at the top of an iteration, before that iteration schedules.
        return bool(self.steps) and self.steps[0].rebalance_drain

    def _maybe_rebalance_kv_pools(self):
        # The pool resize itself is out of scope; the drain is the real one.
        self._consume_previous_batch_for_rebalance()

    def _check_benchmark_disagg_gate(self, scheduled_batch, can_forward):
        return can_forward, False

    def _forward_step(self, scheduled_batch, new_tensors_device=None):
        rows = scheduled_batch.generation_requests
        self.forward_orders.append([req.py_request_id for req in rows])
        if self.step.fail_forward:
            # _handle_errors fails this step's requests and the in-flight ones.
            failed = list(scheduled_batch.all_requests())
            if self.previous_batch is not None:
                failed += self.previous_batch.scheduled_requests.all_requests()
            self.failed.update(req.py_request_id for req in failed)
            return None
        in_flight_seen = None
        for req in rows:
            rid = req.py_request_id
            in_flight = (
                new_tensors_device is not None and not req.is_dummy and req.py_batch_idx is not None
            )
            if in_flight:
                in_flight_seen = rid
                length = self.host_len[rid] + new_tensors_device[req.py_batch_idx]
            else:
                if in_flight_seen is not None:
                    self.errors.append(
                        f"step {self.step_no}: request {rid} is packed from the host "
                        f"after in-flight request {in_flight_seen}"
                    )
                length = self.host_len[rid]
            if length != self.true_len[rid]:
                branch = "in-flight" if in_flight else "host"
                self.errors.append(
                    f"step {self.step_no}: request {rid} packed at length {length} on the "
                    f"{branch} branch, true length {self.true_len[rid]}"
                )
            req.py_batch_idx = req.py_seq_slot
        for req in scheduled_batch.context_requests:
            req.py_batch_idx = req.py_seq_slot
        return {"logits": None}

    def _sample_async(self, scheduled_batch, batch_outputs):
        emitted = {}
        for req in scheduled_batch.context_requests:
            emitted[req] = 1
        for req in scheduled_batch.generation_requests:
            emitted[req] = _new_tokens(self.step_no, req)
        for req, num in emitted.items():
            self.true_len[req.py_request_id] += num
            self.store[req.py_seq_slot] = num
        return SimpleNamespace(device=self.store, emitted=emitted)

    def _update_requests(self, sample_state):
        for req, num in sample_state.emitted.items():
            if req.py_request_id not in self.failed:
                self.host_len[req.py_request_id] += num

    def _take_pending_new_active_requests(self, iter_stats):
        return iter_stats

    @contextlib.contextmanager
    def _profiler(self):
        yield lambda: None

    def _step_scope(self, scheduled_batch, phase=None):
        return contextlib.nullcontext()


for _name in _NOOP_METHODS:
    setattr(_Executor, _name, lambda self, *args, **kwargs: None)


class _NullHangDetector:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def checkpoint(self):
        return None


@pytest.fixture(autouse=True)
def _cpu_loop(monkeypatch):
    """Let the loop's device setup and stream scopes run without a GPU."""
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream=None: contextlib.nullcontext())
    monkeypatch.setattr(py_executor_module, "CUASSERT", lambda result: None)
    monkeypatch.setattr(
        py_executor_module, "cudart", SimpleNamespace(cudaSetDevice=lambda device: (0,))
    )
    monkeypatch.setattr(
        py_executor_module, "ExecutorThreadPin", SimpleNamespace(create=lambda: None)
    )
    monkeypatch.setattr(py_executor_module, "apply_gil_switch_interval", lambda: None)
    monkeypatch.setattr(py_executor_module, "report_default_on_switches", lambda: None)


class _ExitScheduler:
    """A real ``KVCacheV2Scheduler`` over a mocked KV cache manager, set per step
    so that one exit leaves a request out."""

    def __init__(self, monkeypatch, exit_kind: str):
        self.exit_kind = exit_kind
        mgr = Mock()
        mgr.kv_connector_manager = None
        mgr.tokens_per_block = 64
        mgr.enable_block_reuse = False
        mgr.enable_joint_kv_cache_reuse = False
        mgr.enable_partial_reuse = False
        mgr.num_extra_kv_tokens = 0
        mgr._spec_recompute_tail = 0
        mgr.is_draft = False
        mgr.is_vswa = False
        mgr.can_evict = False
        mgr._has_cp_helix = False
        mgr.has_cache_tier_below_gpu = True
        mgr.has_pending_preemption.return_value = False
        mgr.is_request_active.return_value = True
        mgr.try_allocate_generation.return_value = True
        self.peft = Mock(max_device_pages=1)
        self.peft.determine_num_pages.return_value = 1
        with patch(
            "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.KVCacheManagerV2",
            new=type(mgr),
        ):
            self.scheduler = KVCacheV2Scheduler(
                max_batch_size=16,
                max_num_tokens=None,
                kv_cache_manager=mgr,
                scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
                peft_cache_manager=self.peft,
                enable_prefix_aware_scheduling=False,
                enable_recompute_pause=False,
            )

    def schedule(self, active_requests, leave_out: Optional[int]):
        self.scheduler.max_num_tokens = None
        self.peft.max_device_pages = 1
        if leave_out is not None:
            if self.exit_kind == "stop":
                # Room for the two requests ahead of it only: STOP on the token budget.
                self.scheduler.max_num_tokens = 2 * (1 + MAX_DRAFT_LEN)
            else:
                # Its adapter page does not fit: the pass stops at it.
                self.peft.max_device_pages = 0
        out = self.scheduler.schedule_request(active_requests, set())
        scheduled = {req.py_request_id for req in out.generation_requests}
        expected = {req.py_request_id for req in active_requests} - {leave_out}
        assert scheduled == expected, f"the {self.exit_kind} exit did not leave {leave_out} out"
        return out


A, B, X, C, D = 1, 2, 3, 4, 5


def _exit_trace(monkeypatch, exit_kind):
    # STOP and PEFT end the pass at the request, so it goes last.
    order = (A, B, X)
    requests = [_request(rid, lora_task_id=7 if rid == X else None) for rid in order]
    steps = [_Step(), _Step(leave_out=X), _Step()]
    return requests, steps, _ExitScheduler(monkeypatch, exit_kind)


def _scripted(*steps):
    return [_request(rid) for rid in (A, B, X, C, D)], list(steps), None


_SCRIPTED_TRACES = {
    # The group cannot queue at step 3 while this rank has work: the update of
    # step 2 is delayed, so X (in step 2, not in step 3) is still in flight at
    # step 4 while B (left out of step 2) is not.
    "delayed_update": lambda: _scripted(
        _Step(gen=(A, X, B)),
        _Step(gen=(A, X)),
        _Step(gen=(A, B), can_queue=(False, True)),
        _Step(gen=(A, X, B)),
    ),
    # Step 3 fails: its requests and the in-flight ones are failed and the
    # previous batch is dropped. X, left out since step 1, survives.
    "dropped_previous": lambda: _scripted(
        _Step(gen=(A, X)),
        _Step(gen=(A,)),
        _Step(gen=(A, B), fail_forward=True),
        _Step(gen=(D,)),
        _Step(gen=(D, X)),
    ),
    "control_drain": lambda: _scripted(
        _Step(gen=(A, X)),
        _Step(gen=(A,), control_drain=True),
        _Step(gen=(A, X)),
    ),
    "rebalance_drain": lambda: _scripted(
        _Step(gen=(A, X)),
        _Step(gen=(A,), rebalance_drain=True),
        _Step(gen=(A, X)),
    ),
    # Step 3 in scheduler order: A and C in flight (C ran its last context chunk
    # at step 2), X stale, D new. The packer needs X and D first.
    "order": lambda: _scripted(
        _Step(gen=(A, X, B)),
        _Step(gen=(A, B), ctx=(C,)),
        _Step(gen=(A, X, C, D, B)),
    ),
}

_TRACES = ["stop", "peft", *_SCRIPTED_TRACES]


def _run(trace, monkeypatch):
    if trace in _SCRIPTED_TRACES:
        requests, steps, scheduler = _SCRIPTED_TRACES[trace]()
    else:
        requests, steps, scheduler = _exit_trace(monkeypatch, trace)
    executor = _Executor(requests, steps, scheduler=scheduler)
    executor._executor_loop_overlap()
    assert executor.step_no == len(steps) and not executor.steps
    return executor


@pytest.mark.parametrize("trace", _TRACES)
def test_every_row_is_packed_from_the_right_state(trace, monkeypatch):
    executor = _run(trace, monkeypatch)
    assert executor.errors == []


@pytest.mark.parametrize("trace", _TRACES)
def test_the_trace_fails_without_the_clear(trace, monkeypatch):
    monkeypatch.setattr(py_executor_module, "_clear_batch_idx_outside", lambda *args: None)
    executor = _run(trace, monkeypatch)
    assert any("in-flight branch" in error for error in executor.errors), executor.errors


def test_the_clear_precedes_the_generation_request_sort(monkeypatch):
    executor = _run("order", monkeypatch)
    assert executor.forward_orders[-1] == [X, D, A, C, B]
