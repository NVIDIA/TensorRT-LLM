# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chrome filenames through the real profile controller and CPU PyTorch profiler."""

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import profiling

pytestmark = pytest.mark.cpu_only


class _Executor:
    def __init__(self, rank: int = 7) -> None:
        self.iter_counter = 0
        self.global_rank = rank
        self.is_warmup = False
        self.controller = profiling.ProfileController(
            rank=self.global_rank,
            current_iteration=lambda: self.iter_counter,
            is_warmup=lambda: self.is_warmup,
            broadcast_start=lambda config: self.executor_request_queue.enqueue_profile_start_request(
                config
            ),
            broadcast_stop=lambda: self.executor_request_queue.enqueue_profile_stop_request(),
        )


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


@contextmanager
def _profile_step(executor: _Executor) -> Iterator[Callable[[], None]]:
    """Drive the real controller in the executor's end/begin order."""
    with executor.controller.iterations():

        def step() -> None:
            executor.controller.end_iteration()
            executor.controller.begin_iteration()

        yield step


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


@pytest.mark.parametrize("character", ["a", "界"])
def test_long_ranked_trace_paths_fit_filesystem(tmp_path: Path, character: str) -> None:
    """Keep valid names writable and distinct after adding iteration windows."""
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    prefix_bytes = name_max - len(b"-rank-7.json")
    count, padding = divmod(prefix_bytes, len(os.fsencode(character)))
    prefix = character * count + "a" * padding
    outputs = set()
    for rank in (7, 8):
        source = tmp_path / f"{prefix}-rank-{rank}.json"
        source.touch()
        assert profiling._profile_trace_path(str(source), 1, 3, False) == str(source)
        for start, stop in ((1, 3), (5, 7)):
            output = Path(profiling._profile_trace_path(str(source), start, stop, True))
            output.touch(exist_ok=False)
            Path(f"{output}.tmp").touch(exist_ok=False)
            assert output.parent == tmp_path
            assert output.name.endswith(f"-iterations-{start}-{stop}.json")
            assert len(os.fsencode(output.name)) <= name_max
            assert os.fsencode(output.name).decode("utf-8") == output.name
            assert profiling._profile_trace_path(str(source), start, stop, True) == str(output)
            outputs.add(output)
    assert len(outputs) == 4


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("chrome-rank-7.json", "chrome-rank-7-iterations-1-3.json"),
        ("chrome-rank-7", "chrome-rank-7-iterations-1-3"),
        ("chrome-rank-7.trace.json", "chrome-rank-7.trace-iterations-1-3.json"),
    ],
)
def test_fitting_trace_paths_keep_existing_names(
    tmp_path: Path, filename: str, expected: str
) -> None:
    path = f"{tmp_path}//{filename}"
    assert profiling._profile_trace_path(path, 1, 3, True) == f"{tmp_path}//{expected}"
    assert profiling._profile_trace_path(path, 1, 3, False) == path


@pytest.mark.parametrize(("extension", "reserved_bytes"), [(".json", 4), (".gz", 0)])
def test_trace_path_at_byte_limit_keeps_existing_name(
    tmp_path: Path, extension: str, reserved_bytes: int
) -> None:
    prefix = "a" * (
        os.pathconf(tmp_path, "PC_NAME_MAX")
        - reserved_bytes
        - len(os.fsencode(f"-iterations-1-3{extension}"))
    )
    source = str(tmp_path / f"{prefix}{extension}")
    output = profiling._profile_trace_path(source, 1, 3, True)
    assert output == str(tmp_path / f"{prefix}-iterations-1-3{extension}")
    Path(output).touch()


@pytest.mark.parametrize("name_max", [128, -1, None])
def test_trace_path_uses_filesystem_limit_or_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name_max: int | None
) -> None:
    def pathconf(path: str, name: str) -> int:
        assert path == str(tmp_path)
        assert name == "PC_NAME_MAX"
        if name_max is None:
            raise OSError("pathconf unsupported")
        return name_max

    monkeypatch.setattr(os, "pathconf", pathconf)
    output = Path(profiling._profile_trace_path(str(tmp_path / ("a" * 250 + ".json")), 1, 3, True))
    assert len(os.fsencode(output.name)) <= (128 if name_max == 128 else 255)
    output.touch()


def test_trace_path_rejects_extension_that_cannot_fit_window(tmp_path: Path) -> None:
    source = tmp_path / ("a." + "x" * (os.pathconf(tmp_path, "PC_NAME_MAX") - 2))
    source.touch()
    with pytest.raises(ValueError, match="extension and iteration boundaries"):
        profiling._profile_trace_path(str(source), 1, 3, True)


@pytest.mark.parametrize("headroom", [0, 18])
def test_long_environment_paths_export_chrome_traces(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, headroom: int
) -> None:
    """Export when the generated name or only its temporary name exceeds NAME_MAX."""
    prefix = "a" * (os.pathconf(tmp_path, "PC_NAME_MAX") - headroom - len(b"-rank-7.json"))
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3,5-7")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / f"{prefix}.json"))
    executor = _Executor()
    with _profile_step(executor) as step:
        _advance(executor, step, 0, 7)
    traces = list(tmp_path.glob("*.json"))
    assert len(traces) == 2
    for path in traces:
        _event_names(path)


def test_environment_ranges_write_distinct_selected_chrome_traces(
    cpu_runtime: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Export each selected environment window to a distinct iteration path."""
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "1-3,5-7")
    monkeypatch.setenv("TLLM_TORCH_PROFILE_TRACE", str(tmp_path / "chrome.json"))
    executor = _Executor()
    with _profile_step(executor) as step:
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
    with _profile_step(executor) as step:
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
    executor = _Executor(rank=0)
    configs: list[dict] = []
    executor.executor_request_queue = SimpleNamespace(enqueue_profile_start_request=configs.append)
    manager = executor.controller
    expected: set[str] = set()
    with _profile_step(executor) as step:
        for start in (1,) if collision else (1, 4):
            manager.start_profile(output_dir=str(tmp_path), num_steps=2, activities=["CPU"])
            config = configs.pop()
            manager.apply_start_config(config)
            expected.add(f"trtllm-trace-{config['profile_id']}-rank-0.json")
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
    with _profile_step(executor) as step:
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
        with _profile_step(executor) as step:
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
    with _profile_step(executor) as step:
        _advance(executor, step, 0, 3)
    constructor.assert_not_called()
    assert not list(tmp_path.iterdir())
