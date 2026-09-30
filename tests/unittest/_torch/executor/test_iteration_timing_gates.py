# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import profiling, py_executor

pytestmark = pytest.mark.cpu_only

_CASES = [
    (False, 0, True, False, False, True),
    (False, 1, True, False, False, True),
    (True, 0, True, False, False, True),
    (True, 1, True, False, False, False),
    (True, 3, True, False, False, False),
    (True, 1, True, True, False, True),
    (True, 1, True, False, True, True),
    (True, 0, False, False, False, False),
    (True, 1, False, False, False, False),
    (True, 1, False, True, False, False),
    (True, 1, False, False, True, False),
]


class _Executor:
    _should_capture_iter_gpu_timing = py_executor.PyExecutor._should_capture_iter_gpu_timing

    def __init__(self, adp, rank, stats, log, perf) -> None:
        self.enable_attention_dp = adp
        self.dist = SimpleNamespace(rank=rank)
        self.global_rank = rank
        self.enable_iter_perf_stats = stats
        self.print_log = log
        self.perf_manager = Mock(enabled=perf)
        self.perf_manager.borrow_forward_timing_events.return_value = (object(), object())
        self._profile_state_lock = threading.Lock()
        self.profile_start_iters = set()
        self.profile_stop_iters = set()
        self.iter_counter = 0
        self.is_warmup = False
        self._latest_host_step_time_ms = None
        self._latest_prev_device_step_time_ms = None


@pytest.mark.parametrize("adp,rank,stats,log,perf,expected", _CASES)
def test_iteration_gpu_timing_policy(adp, rank, stats, log, perf, expected):
    executor = _Executor(adp, rank, stats, log, perf)
    assert executor._should_capture_iter_gpu_timing() is expected


@pytest.mark.parametrize("adp,rank,stats,log,perf,expected", _CASES)
def test_profile_checkpoint_skips_unused_adp_events(
    monkeypatch, adp, rank, stats, log, perf, expected
):
    executor = _Executor(adp, rank, stats, log, perf)
    calibrator = Mock()
    host_profiler = Mock()
    events = []

    def new_event(**kwargs):
        assert kwargs == {"enable_timing": True}
        event = Mock(elapsed_time=Mock(return_value=0.5))
        events.append(event)
        return event

    event_factory = Mock(side_effect=new_event)
    monkeypatch.setattr(profiling.torch.cuda, "Event", event_factory)
    monkeypatch.setattr(profiling, "get_calibrator", lambda: calibrator)
    monkeypatch.setattr(profiling, "get_global_profiler", lambda: host_profiler)
    monkeypatch.delenv(profiling.PROFILE_TRACE_ENV_VAR_NAME, raising=False)
    monkeypatch.setenv(profiling.PROFILE_LOG_RANKS_ENV_VAR_NAME, "999")
    manager = profiling.PyExecutorProfileManager(executor)
    with manager.profile_step() as step:
        for _ in range(4):
            step()
            executor.iter_counter += 1

    assert calibrator.pre_step.call_count == 4
    assert calibrator.post_step.call_count == 4
    assert host_profiler.notify_iteration.call_count == 4
    if log or expected:
        assert event_factory.call_count == 4
        assert executor._latest_prev_device_step_time_ms == 0.5
    else:
        assert sum(event.record.call_count for event in events) == 0
        assert sum(event.query.call_count for event in events) == 0
        assert sum(event.synchronize.call_count for event in events) == 0
        assert executor._latest_prev_device_step_time_ms is None
    assert (executor._latest_host_step_time_ms is not None) == (stats or log)


@pytest.fixture(scope="module")
def forward_event_statements():
    """Execute each loop's event-acquisition branch without a model forward."""
    tree = ast.parse(Path(py_executor.__file__).read_text())
    executor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PyExecutor"
    )
    statements = {}
    for method in executor_class.body:
        if not isinstance(method, ast.FunctionDef) or not method.name.startswith("_executor_loop"):
            continue
        for node in ast.walk(method):
            if not isinstance(node, ast.If):
                continue
            if any(
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "borrow_forward_timing_events"
                for statement in node.body
            ):
                statements[method.name] = compile(
                    ast.Module(body=[node], type_ignores=[]), py_executor.__file__, "exec"
                )
    assert set(statements) == {"_executor_loop", "_executor_loop_overlap", "_executor_loop_pp"}
    return statements


@pytest.mark.parametrize(
    "loop_name", ["_executor_loop", "_executor_loop_overlap", "_executor_loop_pp"]
)
@pytest.mark.parametrize("adp,rank,stats,log,perf,expected", _CASES)
def test_all_executor_loops_skip_unused_forward_events(
    forward_event_statements, loop_name, adp, rank, stats, log, perf, expected
):
    executor = _Executor(adp, rank, stats, log, perf)
    scope = {
        "self": executor,
        "gpu_forward_start": None,
        "gpu_forward_end": None,
        "gpu_forward_events_from_perf_pool": False,
    }
    exec(forward_event_statements[loop_name], scope)
    assert executor.perf_manager.borrow_forward_timing_events.call_count == int(expected)
    assert scope["gpu_forward_events_from_perf_pool"] is expected


@pytest.mark.parametrize("loop_name", ["_executor_loop", "_executor_loop_overlap"])
def test_shared_request_performance_events_are_not_replaced(forward_event_statements, loop_name):
    executor = _Executor(True, 1, True, False, True)
    forward_start, forward_end = object(), object()
    scope = {
        "self": executor,
        "gpu_forward_start": forward_start,
        "gpu_forward_end": forward_end,
        "gpu_forward_events_from_perf_pool": False,
    }
    exec(forward_event_statements[loop_name], scope)
    executor.perf_manager.borrow_forward_timing_events.assert_not_called()
    assert scope["gpu_forward_start"] is forward_start
    assert scope["gpu_forward_end"] is forward_end
