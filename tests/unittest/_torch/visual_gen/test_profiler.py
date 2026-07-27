# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, call, patch

import pytest
import torch

import tensorrt_llm._torch.visual_gen
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.profiler import (
    PROFILE_START_STOP_ENV_VAR_NAME,
    PROFILE_TRACE_ENV_VAR_NAME,
    VisualGenProfiler,
)


def _profiler_with_torch_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_range: str,
    rank: int = 0,
) -> Tuple[VisualGenProfiler, MagicMock]:
    """Build a profiler whose torch.profiler session is a mock."""
    monkeypatch.setenv(PROFILE_START_STOP_ENV_VAR_NAME, profile_range)
    monkeypatch.setenv(PROFILE_TRACE_ENV_VAR_NAME, str(tmp_path / "visual-gen-trace.json"))
    torch_profiler = MagicMock()
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.profiler, "profile", return_value=torch_profiler),
    ):
        profiler = VisualGenProfiler(rank=rank)
    return profiler, torch_profiler


def test_setup_torch_profiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(PROFILE_START_STOP_ENV_VAR_NAME, "0-4")
    monkeypatch.setenv(PROFILE_TRACE_ENV_VAR_NAME, str(tmp_path / "visual-gen-trace.json"))
    torch_profiler = MagicMock()

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.profiler, "profile", return_value=torch_profiler) as profile,
    ):
        profiler = VisualGenProfiler(rank=2)

    assert profiler.range == (frozenset({0}), frozenset({4}))
    assert profiler._torch_profiler is torch_profiler
    assert profiler._trace_path == str(tmp_path / "visual-gen-trace-rank-2.json")
    profile.assert_called_once_with(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    )


def test_setup_torch_profiler_requires_profile_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(PROFILE_START_STOP_ENV_VAR_NAME, raising=False)
    monkeypatch.setenv(PROFILE_TRACE_ENV_VAR_NAME, str(tmp_path / "visual-gen-trace.json"))

    with (
        patch.object(torch.profiler, "profile") as profile,
        patch("tensorrt_llm._torch.visual_gen.profiler.logger.warning") as warning,
    ):
        profiler = VisualGenProfiler()

    assert not profiler.enabled
    profile.assert_not_called()
    warning.assert_called_once()
    assert PROFILE_START_STOP_ENV_VAR_NAME in warning.call_args.args[0]


def test_window_controls_torch_profiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profiler, torch_profiler = _profiler_with_torch_trace(monkeypatch, tmp_path, "0-4")
    cudart = MagicMock()

    with (
        patch("tensorrt_llm._torch.visual_gen.profiler.torch.cuda.cudart", return_value=cudart),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "synchronize"),
        patch("tensorrt_llm._torch.visual_gen.profiler.logger.info"),
    ):
        profiler.open_window()
        profiler.close_window()

    cudart.cudaProfilerStart.assert_called_once_with()
    torch_profiler.start.assert_called_once_with()
    torch_profiler.stop.assert_called_once_with()
    torch_profiler.export_chrome_trace.assert_called_once_with(
        str(tmp_path / "visual-gen-trace-rank-0.json")
    )
    cudart.cudaProfilerStop.assert_called_once_with()
    assert not profiler.active
    assert profiler._torch_profiler is None


def test_close_window_syncs_before_ending_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A collector may end the process the moment the capture range closes.

    ``nsys --capture-range-end=stop-shutdown`` does exactly that, so the
    device must be idle before ``cudaProfilerStop()``.
    """
    profiler, torch_profiler = _profiler_with_torch_trace(monkeypatch, tmp_path, "all")
    order = MagicMock()
    cudart = MagicMock()

    with (
        patch("tensorrt_llm._torch.visual_gen.profiler.torch.cuda.cudart", return_value=cudart),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "synchronize") as synchronize,
        patch("tensorrt_llm._torch.visual_gen.profiler.logger.info"),
    ):
        profiler.open_window()
        order.attach_mock(synchronize, "synchronize")
        order.attach_mock(torch_profiler.stop, "torch_stop")
        order.attach_mock(cudart.cudaProfilerStop, "cudaProfilerStop")
        profiler.close_window()

    assert order.mock_calls == [
        call.synchronize(),
        call.torch_stop(),
        call.cudaProfilerStop(),
    ]


def test_close_window_is_a_noop_when_never_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profiler, torch_profiler = _profiler_with_torch_trace(monkeypatch, tmp_path, "all")
    cudart = MagicMock()

    with (
        patch("tensorrt_llm._torch.visual_gen.profiler.torch.cuda.cudart", return_value=cudart),
        patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        profiler.close_window()

    synchronize.assert_not_called()
    cudart.cudaProfilerStop.assert_not_called()
    torch_profiler.stop.assert_not_called()


def test_each_window_uses_a_fresh_trace_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profiler, first_profiler = _profiler_with_torch_trace(monkeypatch, tmp_path, "0-1,2-3")
    second_profiler = MagicMock()
    cudart = MagicMock()

    with (
        patch.object(torch.profiler, "profile", return_value=second_profiler) as profile,
        patch("tensorrt_llm._torch.visual_gen.profiler.torch.cuda.cudart", return_value=cudart),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "synchronize"),
        patch("tensorrt_llm._torch.visual_gen.profiler.logger.info"),
    ):
        profiler.open_window()
        profiler.close_window()
        profiler.open_window()
        profiler.close_window()

    profile.assert_called_once()
    first_profiler.export_chrome_trace.assert_called_once_with(
        str(tmp_path / "visual-gen-trace-rank-0.json")
    )
    second_profiler.export_chrome_trace.assert_called_once_with(
        str(tmp_path / "visual-gen-trace-rank-0-window-1.json")
    )
    assert cudart.cudaProfilerStart.call_count == 2
    assert cudart.cudaProfilerStop.call_count == 2


def test_cuda_gate_closes_when_trace_export_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profiler, torch_profiler = _profiler_with_torch_trace(monkeypatch, tmp_path, "all")
    torch_profiler.export_chrome_trace.side_effect = RuntimeError("export failed")
    cudart = MagicMock()

    with (
        patch("tensorrt_llm._torch.visual_gen.profiler.torch.cuda.cudart", return_value=cudart),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "synchronize"),
        patch("tensorrt_llm._torch.visual_gen.profiler.logger.info"),
    ):
        profiler.open_window()
        with pytest.raises(RuntimeError, match="export failed"):
            profiler.close_window()

    cudart.cudaProfilerStop.assert_called_once_with()
    assert not profiler.active
    assert profiler._torch_profiler is None


def _recording_profiler(
    monkeypatch: pytest.MonkeyPatch, profile_range: str
) -> Tuple[VisualGenProfiler, List[str]]:
    """A profiler whose window primitives record instead of driving CUPTI."""
    monkeypatch.setenv(PROFILE_START_STOP_ENV_VAR_NAME, profile_range)
    monkeypatch.delenv(PROFILE_TRACE_ENV_VAR_NAME, raising=False)
    profiler = VisualGenProfiler()
    events: List[str] = []

    def open_window() -> None:
        if not profiler._active:
            events.append("start")
            profiler._active = True

    def close_window() -> None:
        if profiler._active:
            events.append("stop")
            profiler._active = False

    monkeypatch.setattr(profiler, "open_window", open_window)
    monkeypatch.setattr(profiler, "close_window", close_window)
    return profiler, events


def _drive_request(profiler: VisualGenProfiler, events: List[str], num_steps: int = 0) -> None:
    """Replay the boundaries a pipeline hits over one request."""
    with profiler.request_scope():
        events.append("text_encode")
        for i, _ in profiler.steps(range(num_steps)):
            events.append(f"step{i}")
        events.append("vae_decode")


@pytest.mark.parametrize(
    ("profile_range", "expected_events"),
    [
        ("all", ["start", "text_encode", "step0", "vae_decode", "stop"]),
        ("predenoise", ["start", "text_encode", "stop", "step0", "vae_decode"]),
        ("postdenoise", ["text_encode", "step0", "start", "vae_decode", "stop"]),
    ],
)
def test_request_scope_owns_phase_boundaries(
    monkeypatch: pytest.MonkeyPatch, profile_range: str, expected_events: List[str]
) -> None:
    profiler, events = _recording_profiler(monkeypatch, profile_range)

    _drive_request(profiler, events, num_steps=1)

    assert events == expected_events
    assert not profiler.active


def test_request_scope_closes_window_when_inference_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler, events = _recording_profiler(monkeypatch, "0-4")

    with pytest.raises(RuntimeError, match="boom"):
        with profiler.request_scope():
            for _ in profiler.steps(range(4)):
                raise RuntimeError("boom")

    assert events == ["start", "stop"]
    assert not profiler.active


def test_single_shot_phases_do_not_rearm(monkeypatch: pytest.MonkeyPatch) -> None:
    profiler, events = _recording_profiler(monkeypatch, "predenoise")

    _drive_request(profiler, events, num_steps=1)
    events.clear()
    _drive_request(profiler, events, num_steps=1)

    assert events == ["text_encode", "step0", "vae_decode"]


def test_step_windows_follow_numeric_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    profiler, events = _recording_profiler(monkeypatch, "1-2")

    with profiler.request_scope():
        for i, _ in profiler.steps(range(4)):
            events.append(f"step{i}")

    assert events == ["step0", "start", "step1", "step2", "stop", "step3"]


class _StubPipeline:
    """Minimal stand-in exercising ``BasePipeline.run_inference`` glue."""

    run_inference = BasePipeline.run_inference

    def __init__(self, profiler: VisualGenProfiler, is_warmup: bool = False) -> None:
        self._profiler = profiler
        self._is_warmup = is_warmup

    def infer(self, req: object) -> object:
        return req


def test_run_inference_skips_profiling_during_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler, events = _recording_profiler(monkeypatch, "all")
    pipeline = _StubPipeline(profiler, is_warmup=True)

    request = object()
    assert pipeline.run_inference(request) is request
    assert events == []
    assert not profiler.active

    pipeline._is_warmup = False
    assert pipeline.run_inference(request) is request
    assert events == ["start", "stop"]


# ----------------------------------------------------------------------
# Structural guard: a hand-written denoise loop that forgets the hooks is
# silently absent from every trace, which is how Qwen-Image and LTX-2
# stage 2 were missed. Catch it at the source level instead.
# ----------------------------------------------------------------------

_VISUAL_GEN_ROOT = Path(tensorrt_llm._torch.visual_gen.__file__).parent

_PROFILE_STEPS_HOOK = "_profile_denoise_steps"


def _self_calls(node: ast.AST) -> set:
    """Names of ``self.<name>(...)`` calls anywhere under ``node``."""
    return {
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    }


def _denoise_loops() -> List[Tuple[str, str, ast.For, ast.AST]]:
    """Every loop in visual_gen that steps a transformer, i.e. denoises."""
    found = []
    for path in sorted(_VISUAL_GEN_ROOT.rglob("*.py")):
        for fn in ast.walk(ast.parse(path.read_text())):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for loop in ast.walk(fn):
                if not isinstance(loop, ast.For):
                    continue
                names = _self_calls(loop)
                if "transformer" in names or any(n.startswith("_denoise_step") for n in names):
                    found.append((path.name, fn.name, loop, fn))
    return found


def test_every_denoise_loop_is_instrumented() -> None:
    loops = _denoise_loops()
    # base denoise(), Qwen-Image forward(), LTX-2 stage 2 _refinement_denoise()
    assert len(loops) == 3, [(f, n) for f, n, _, _ in loops]

    for filename, fn_name, loop, _ in loops:
        assert _PROFILE_STEPS_HOOK in _self_calls(loop.iter), (
            f"{filename}::{fn_name} steps a transformer over a plain iterator; "
            f"use self.{_PROFILE_STEPS_HOOK}() so the loop appears in "
            "TLLM_PROFILE_VISUAL_GEN_START_STOP traces"
        )
