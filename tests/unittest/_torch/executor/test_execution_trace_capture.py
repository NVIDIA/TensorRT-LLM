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
"""Execution capture through the real profile manager, without a model or GPU."""

import json
import tempfile
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
    """Only the executor state consumed by the profile manager."""

    def __init__(self, ranges: str = "1-3", rank: int = 7) -> None:
        spans = [tuple(map(int, span.split("-"))) for span in ranges.split(",")]
        self.profile_start_iters = {start for start, _ in spans}
        self.profile_stop_iters = {stop for _, stop in spans}
        self.iter_counter = 0
        self.global_rank = rank
        self.is_warmup = False
        self.print_log = False
        self.enable_iter_perf_stats = False
        self._runtime_profile_trace_path: str | None = None
        self._runtime_profile_activities: list[str] | None = None
        self._runtime_profile_cuda_only = False
        self._runtime_profile_pending_start_iter: int | None = None
        self._runtime_profile_pending_stop_iter: int | None = None
        self._profile_state_lock = threading.Lock()
        self._profile_enabled = False


@pytest.fixture
def cpu_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Mock]:
    """Replace unavailable CUDA timing/API calls, not the PyTorch profiler."""
    for name in (
        "TLLM_PROFILE_START_STOP",
        "TLLM_TORCH_PROFILE_TRACE",
        "ENABLE_PYTORCH_EXECUTION_TRACE",
        "ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "Event", _CpuEvent)
    original_profile = torch.profiler.profile

    def cpu_profile(
        *,
        activities: list[torch.profiler.ProfilerActivity],
        schedule: Callable[[int], torch.profiler.ProfilerAction],
        record_shapes: bool,
        with_modules: bool,
        experimental_config: torch._C._profiler._ExperimentalConfig,
        execution_trace_observer: torch.profiler.ExecutionTraceObserver | None = None,
    ) -> torch.profiler.profile:
        return original_profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=schedule,
            record_shapes=record_shapes,
            with_modules=with_modules,
            experimental_config=experimental_config,
            execution_trace_observer=execution_trace_observer,
        )

    monkeypatch.setattr(torch.profiler, "profile", cpu_profile)
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: SimpleNamespace(cudaProfilerStart=lambda: None, cudaProfilerStop=lambda: None),
    )
    calibrator = Mock()
    monkeypatch.setattr(profiling, "get_calibrator", lambda: calibrator)
    monkeypatch.setattr(profiling, "get_global_profiler", lambda: None)
    yield calibrator


def _execution_trace(directory: Path) -> dict:
    paths = list(directory.glob("*.et.json"))
    assert len(paths) == 1, paths
    return json.loads(paths[0].read_text())


def _run(executor: _Executor, stop: int = 3) -> None:
    with profiling.PyExecutorProfileManager(executor).profile_step() as step:
        for index in range(stop + 1):
            executor.iter_counter = index
            step()
            with torch.profiler.record_function(f"iteration-{index}"):
                torch.ones(2) + 1


@pytest.mark.parametrize("chrome", [False, True])
def test_pytorch_flag_captures_selected_steps(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chrome: bool
) -> None:
    # Use PyTorch output creation and capture only the selected executor steps.
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    if chrome:
        monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    _run(_Executor())
    execution = _execution_trace(tmp_path)
    names = {node["name"] for node in execution["nodes"]}
    assert "aten::add" in names
    assert any(name.startswith("ProfilerStep#") for name in names)
    assert {"iteration-1", "iteration-2"} <= names
    assert not {"iteration-0", "iteration-3"} & names
    if chrome:
        trace = json.loads((tmp_path / "chrome-rank-7.json").read_text())
        assert any(event.get("name") == "aten::add" for event in trace["traceEvents"])
    else:
        assert not list(tmp_path.glob("chrome*"))


def test_multiple_ranges_write_distinct_real_traces(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Chrome ranges must not overwrite; the observer aggregates both ranges.
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-100,400-500")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    _run(_Executor("1-100,400-500"), stop=500)
    for start, stop in ((1, 100), (400, 500)):
        path = tmp_path / f"chrome-rank-7-iterations-{start}-{stop}.json"
        trace = json.loads(path.read_text())
        assert any(record.get("name") == "aten::add" for record in trace["traceEvents"])
    execution = _execution_trace(tmp_path)
    assert sum(node["name"] == "aten::add" for node in execution["nodes"]) == 199
    assert len(list(tmp_path.glob("*.et.json"))) == 1
    assert not (tmp_path / "chrome-rank-7.json").exists()


def test_execution_capture_requires_iteration_ranges(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The PyTorch flag alone must not start executor profiling without a window.
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    _run(_Executor())
    assert not list(tmp_path.iterdir())


def test_warmup_skipped_start_does_not_capture(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    executor = _Executor()
    executor._runtime_profile_pending_start_iter = 1
    executor._runtime_profile_pending_stop_iter = 3
    with profiling.PyExecutorProfileManager(executor).profile_step() as step:
        executor.iter_counter = 1
        executor.is_warmup = True
        step()
        executor.iter_counter = 3
        executor.is_warmup = False
        step()
    assert not list(tmp_path.iterdir())
    assert not executor._profile_enabled
    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_pending_stop_iter is None


@pytest.mark.parametrize("body_error", [False, True])
def test_early_exit_flushes_and_removes_observer(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body_error: bool
) -> None:
    # Finalization must flush live capture; a leaked callback blocks the next observer.
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    executor = _Executor()
    error = ValueError("workload failed")
    try:
        with profiling.PyExecutorProfileManager(executor).profile_step() as step:
            executor.iter_counter = 1
            step()
            torch.ones(2) + 1
            if body_error:
                raise error
    except ValueError as caught:
        assert body_error and caught is error
    trace = _execution_trace(tmp_path)
    assert any(node["name"] == "aten::add" for node in trace["nodes"])
    assert not executor._profile_enabled
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "next.json")
    )
    try:
        assert observer.is_registered
    finally:
        observer.unregister_callback()


@pytest.mark.parametrize("failure", ["construct", "start", "stop", "export", "unregister"])
def test_capture_failure_releases_all_resources_and_preserves_body_error(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    # Rollback/finalization must release the callback and state even if a resource fails.
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    observer = Mock()
    profiler = Mock()
    profiler.execution_trace_observer = observer
    profiler.start.side_effect = observer.start
    profiler.stop.side_effect = observer.stop
    failure_error = RuntimeError("capture failed")
    if failure == "construct":
        monkeypatch.setattr(torch.profiler, "profile", Mock(side_effect=failure_error))
    else:
        monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: profiler)
        method = {
            "start": profiler.start,
            "stop": profiler.stop,
            "export": profiler.export_chrome_trace,
            "unregister": observer.unregister_callback,
        }[failure]
        method.side_effect = failure_error
    executor = _Executor()
    executor._runtime_profile_pending_start_iter = 1
    executor._runtime_profile_pending_stop_iter = 3
    body_error = ValueError("original workload failure")
    with pytest.raises((RuntimeError, ValueError)) as caught:
        with profiling.PyExecutorProfileManager(executor).profile_step() as step:
            executor.iter_counter = 1
            step()
            raise body_error
    assert caught.value is (failure_error if failure in ("construct", "start") else body_error)
    assert not executor._profile_enabled
    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_pending_stop_iter is None
    if failure == "construct":
        observer.unregister_callback.assert_not_called()
    else:
        observer.unregister_callback.assert_called_once()
    cpu_runtime.stop.assert_called_once()


@pytest.mark.parametrize("failure", ["start", "stop"])
def test_failed_profiler_window_stops_observer_before_context_exit(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    # Catching a window failure must not capture subsequent executor work.
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    start_error = RuntimeError("profiler failed after starting observer")
    stop_error = RuntimeError("profiler failed before stopping observer")

    def failing_profile(
        *, execution_trace_observer: torch.profiler.ExecutionTraceObserver | None = None, **kwargs
    ) -> SimpleNamespace:
        if execution_trace_observer is None:
            execution_trace_observer = (
                torch.profiler.ExecutionTraceObserver.build_execution_trace_obs_from_env()
            )

        def start() -> None:
            if execution_trace_observer is not None:
                execution_trace_observer.start()
            if failure == "start":
                raise start_error

        def stop() -> None:
            raise stop_error

        return SimpleNamespace(
            execution_trace_observer=execution_trace_observer,
            start=start,
            stop=stop,
            step=lambda: None,
        )

    monkeypatch.setattr(torch.profiler, "profile", failing_profile)
    executor = _Executor()
    with profiling.PyExecutorProfileManager(executor).profile_step() as step:
        executor.iter_counter = 1
        with pytest.raises(RuntimeError) as caught:
            step()
            executor.iter_counter = 3
            step()
        assert caught.value is (start_error if failure == "start" else stop_error)
        assert not executor._profile_enabled
        with torch.profiler.record_function("after-failed-window"):
            torch.ones(2) + 1
    execution = _execution_trace(tmp_path)
    assert "after-failed-window" not in {node["name"] for node in execution["nodes"]}
    cpu_runtime.stop.assert_called_once()
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "next.json")
    )
    try:
        assert observer.is_registered
    finally:
        observer.unregister_callback()


def test_cleanup_error_without_body_error_is_reported(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    error = RuntimeError("unregistration failed")
    original = torch.profiler.ExecutionTraceObserver.unregister_callback

    def fail_after_unregister(observer: torch.profiler.ExecutionTraceObserver) -> None:
        was_registered = observer.is_registered
        original(observer)
        if was_registered:
            raise error

    monkeypatch.setattr(
        torch.profiler.ExecutionTraceObserver, "unregister_callback", fail_after_unregister
    )
    executor = _Executor()
    with pytest.raises(RuntimeError) as caught:
        _run(executor)
    assert caught.value is error
    assert not executor._profile_enabled


def test_scheduled_stop_failure_clears_runtime_state(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    error = RuntimeError("calibrator stop failed")
    cpu_runtime.stop.side_effect = error
    executor = _Executor()
    executor._runtime_profile_pending_start_iter = 1
    executor._runtime_profile_pending_stop_iter = 3
    with pytest.raises(RuntimeError) as caught:
        _run(executor)
    assert caught.value is error
    assert not executor._profile_enabled
    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_pending_stop_iter is None
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "next.json")
    )
    try:
        assert observer.is_registered
    finally:
        observer.unregister_callback()


@pytest.mark.parametrize("resource", ["calibrator", "cuda"])
def test_failed_start_before_observer_registration_releases_runtime_claim(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resource: str
) -> None:
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    error = RuntimeError("resource startup failed")
    if resource == "calibrator":
        cpu_runtime.start.side_effect = error
    else:
        monkeypatch.setattr(
            torch.cuda,
            "cudart",
            lambda: SimpleNamespace(
                cudaProfilerStart=Mock(side_effect=error), cudaProfilerStop=lambda: None
            ),
        )
    executor = _Executor()
    executor._runtime_profile_pending_start_iter = 1
    executor._runtime_profile_pending_stop_iter = 3
    with pytest.raises(RuntimeError) as caught:
        _run(executor)
    assert caught.value is error
    assert not list(tmp_path.iterdir())
    assert not executor._profile_enabled
    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_pending_stop_iter is None


def test_body_error_before_start_releases_pending_runtime_window(
    cpu_runtime: Mock,
) -> None:
    executor = _Executor()
    executor._runtime_profile_pending_start_iter = 1
    executor._runtime_profile_pending_stop_iter = 3
    error = ValueError("executor exited before scheduled start")
    with pytest.raises(ValueError) as caught:
        with profiling.PyExecutorProfileManager(executor).profile_step():
            raise error
    assert caught.value is error
    assert executor._runtime_profile_pending_start_iter is None
    assert executor._runtime_profile_pending_stop_iter is None


@pytest.mark.parametrize("enabled", [None, "0", "false"])
def test_chrome_only_does_not_register_execution_observer(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enabled: str | None
) -> None:
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    if enabled is not None:
        monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", enabled)
    _run(_Executor())
    assert (tmp_path / "chrome-rank-7.json").is_file()
    assert not list(tmp_path.glob("*.et.json"))


@pytest.mark.parametrize("cuda_only", [False, True])
def test_runtime_windows_reuse_pytorch_observer_without_env_ranges(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cuda_only: bool
) -> None:
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    executor = _Executor()
    executor.profile_start_iters.clear()
    executor.profile_stop_iters.clear()
    manager = profiling.PyExecutorProfileManager(executor)
    with manager.profile_step() as step:
        for index in range(8):
            executor.iter_counter = index
            if index in (0, 4):
                manager.apply_start_config(
                    {
                        "output_dir": str(tmp_path),
                        "profile_id": f"runtime-{index}",
                        "activities": ["CUDA_PROFILER"] if cuda_only else ["CPU"],
                        "num_steps": 2,
                    }
                )
            step()
            with torch.profiler.record_function(f"iteration-{index}"):
                torch.ones(2) + 1
    if cuda_only:
        assert not list(tmp_path.iterdir())
    else:
        trace = _execution_trace(tmp_path)
        names = {node["name"] for node in trace["nodes"]}
        assert {"iteration-1", "iteration-2", "iteration-5", "iteration-6"} <= names
        assert not {"iteration-0", "iteration-3", "iteration-4", "iteration-7"} & names
        assert len(list(tmp_path.glob("trtllm-trace-runtime-*-rank-7.json"))) == 2


def test_pytorch_execution_trace_extras_are_preserved(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE", "1")
    monkeypatch.setenv("ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS", "1")
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    _run(_Executor())
    assert any(node["name"] == "aten::add" for node in _execution_trace(tmp_path)["nodes"])
    assert len(list(tmp_path.glob("*.et_resources"))) == 1


def test_removed_trt_execution_flag_does_not_enable_capture(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TLLM_TORCH_EXECUTION_TRACE", str(tmp_path / "execution.json"))
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3")
    _run(_Executor())
    assert not list(tmp_path.iterdir())
