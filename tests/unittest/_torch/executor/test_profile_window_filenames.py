# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chrome filenames through the real profile manager and CPU PyTorch profiler."""

import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import profiling

pytestmark = pytest.mark.cpu_only

if issubclass(torch.cuda.Event, torch.Event):
    _CpuEventBase = torch.Event
else:
    from torch._streambase import _EventBase as _CpuEventBase


class _CpuEvent(_CpuEventBase):
    def __new__(cls, enable_timing: bool = False) -> "_CpuEvent":
        """Create a CPU event compatible with the installed PyTorch event base."""
        if issubclass(cls, torch.Event):
            return torch.Event.__new__(cls, device="cpu", enable_timing=enable_timing)
        return super().__new__(cls)

    def __init__(self, enable_timing: bool = False) -> None:
        pass

    def record(self, stream: object = None) -> None:
        pass

    def wait(self, stream: object = None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class _Executor:
    def __init__(self) -> None:
        starts, stops = profiling.load_iteration_indexes("TLLM_PROFILE_START_STOP")
        self.profile_start_iters = set(starts)
        self.profile_stop_iters = set(stops)
        self.iter_counter = 0
        self.model_engine = SimpleNamespace(model=None)
        self.global_rank = 7
        self.is_warmup = False
        self.print_log = False
        self.enable_iter_perf_stats = False
        self._runtime_profile_trace_path = None
        self._runtime_profile_activities = None
        self._runtime_profile_cuda_only = False
        self._runtime_profile_pending_start_iter = None
        self._runtime_profile_pending_stop_iter = None
        self._profile_state_lock = threading.Lock()
        self._profile_enabled = False


@pytest.fixture
def cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    """Replace CUDA boundaries while retaining real CPU trace generation."""
    for name in (
        "TLLM_PROFILE_START_STOP",
        "TLLM_TORCH_PROFILE_TRACE",
        "ENABLE_PYTORCH_EXECUTION_TRACE",
        "ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS",
    ):
        monkeypatch.delenv(name, raising=False)
    profiling.load_iteration_indexes.cache_clear()
    monkeypatch.setattr(torch.cuda, "Event", _CpuEvent)
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: SimpleNamespace(cudaProfilerStart=lambda: None, cudaProfilerStop=lambda: None),
    )
    monkeypatch.setattr(profiling, "get_calibrator", lambda: Mock())
    monkeypatch.setattr(profiling, "get_global_profiler", lambda: None)
    original_profile = torch.profiler.profile

    def cpu_profile(**options: object) -> torch.profiler.profile:
        options["activities"] = [torch.profiler.ProfilerActivity.CPU]
        return original_profile(**options)

    create = Mock(side_effect=cpu_profile)
    monkeypatch.setattr(torch.profiler, "profile", create)
    try:
        yield create
    finally:
        profiling.load_iteration_indexes.cache_clear()


def _advance(executor: _Executor, step: Callable[[], None], first: int, last: int) -> None:
    """Advance executor iterations and emit identifiable CPU work."""
    for index in range(first, last + 1):
        executor.iter_counter = index
        step()
        with torch.profiler.record_function(f"iteration-{index}"):
            torch.ones(2) + 1


def _event_names(path: Path) -> set[str]:
    """Read Chrome event names and require the recorded tensor operation."""
    names = {event["name"] for event in json.loads(path.read_text())["traceEvents"]}
    assert "aten::add" in names
    return names


def test_environment_ranges_write_distinct_selected_chrome_traces(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Export each selected environment window to a distinct iteration path."""
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3,5-7")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    executor = _Executor()
    with profiling.PyExecutorProfileManager(executor).profile_step() as step:
        _advance(executor, step, 0, 7)
    assert {path.name for path in tmp_path.glob("*.json")} == {
        "chrome-rank-7-iterations-1-3.json",
        "chrome-rank-7-iterations-5-7.json",
    }
    for start, stop in ((1, 3), (5, 7)):
        names = _event_names(tmp_path / f"chrome-rank-7-iterations-{start}-{stop}.json")
        assert {f"iteration-{start}", f"iteration-{start + 1}"} <= names
        other_iterations = {
            f"iteration-{index}" for index in range(8) if index not in (start, start + 1)
        }
        assert other_iterations.isdisjoint(names)


def test_single_environment_range_keeps_rank_filename(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep the existing rank-only filename for one environment window."""
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    executor = _Executor()
    with profiling.PyExecutorProfileManager(executor).profile_step() as step:
        _advance(executor, step, 0, 3)
    assert {path.name for path in tmp_path.glob("*.json")} == {"chrome-rank-7.json"}
    assert {"iteration-1", "iteration-2"} <= _event_names(tmp_path / "chrome-rank-7.json")


@pytest.mark.parametrize("collision", [False, True])
def test_runtime_api_keeps_unique_paths_and_requested_configuration(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, collision: bool
) -> None:
    """Preserve runtime paths and options with or without range collisions."""
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3,20-22" if collision else "20-22,30-32")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "environment.json"))
    executor = _Executor()
    configs: list[dict] = []
    executor.executor_request_queue = SimpleNamespace(enqueue_profile_start_request=configs.append)
    manager = profiling.PyExecutorProfileManager(executor)
    expected: set[str] = set()
    with manager.profile_step() as step:
        for start in (1,) if collision else (1, 4):
            manager.start_profile(output_dir=str(tmp_path), num_steps=2, activities=["CPU"])
            config = configs.pop()
            manager.apply_start_config(config)
            expected.add(f"trtllm-trace-{config['profile_id']}-rank-7.json")
            _advance(executor, step, start, start + 2)
    assert len(expected) == (1 if collision else 2)
    assert {path.name for path in tmp_path.glob("*.json")} == expected
    for name in expected:
        _event_names(tmp_path / name)
    assert all(
        call.kwargs["activities"] == [torch.profiler.ProfilerActivity.CPU]
        for call in cpu_runtime.call_args_list
    )


def test_early_exit_filename_uses_actual_end_iteration(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Label an interrupted environment window with its actual final iteration."""
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-10,20-30")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    executor = _Executor()
    with profiling.PyExecutorProfileManager(executor).profile_step() as step:
        _advance(executor, step, 0, 4)
    assert {path.name for path in tmp_path.glob("*.json")} == {"chrome-rank-7-iterations-1-4.json"}
    assert "iteration-4" in _event_names(tmp_path / "chrome-rank-7-iterations-1-4.json")


def test_execution_flag_preserves_stock_profiler_constructor_options(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    constructor = Mock(side_effect=RuntimeError("constructor boundary reached"))
    monkeypatch.setattr(torch.profiler, "profile", constructor)
    executor = _Executor()
    with pytest.raises(RuntimeError, match="constructor boundary reached"):
        with profiling.PyExecutorProfileManager(executor).profile_step() as step:
            _advance(executor, step, 0, 1)
    constructor.assert_called_once_with(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.XPU,
        ],
        record_shapes=True,
        with_modules=True,
    )


def test_execution_flag_and_ranges_without_chrome_path_do_not_start_capture(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    constructor = Mock(side_effect=AssertionError("capture must remain disabled"))
    monkeypatch.setattr(torch.profiler, "profile", constructor)
    executor = _Executor()
    with profiling.PyExecutorProfileManager(executor).profile_step() as step:
        _advance(executor, step, 0, 3)
    constructor.assert_not_called()
    assert not list(tmp_path.iterdir())
