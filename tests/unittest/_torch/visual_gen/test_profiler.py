# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.visual_gen.pipeline import (
    PROFILE_START_STOP_ENV_VAR_NAME,
    PROFILE_TRACE_ENV_VAR_NAME,
    BasePipeline,
)


def test_setup_torch_profiler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROFILE_TRACE_ENV_VAR_NAME, "/tmp/visual-gen-trace.json")
    pipeline = SimpleNamespace(
        _profile_range=(frozenset({0}), frozenset({4})),
        _torch_profiler=None,
        _torch_profile_trace_path=None,
        rank=2,
    )
    pipeline._create_torch_profiler = MethodType(BasePipeline._create_torch_profiler, pipeline)
    torch_profiler = MagicMock()

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.profiler, "profile", return_value=torch_profiler) as profile,
    ):
        BasePipeline._setup_torch_profiler(pipeline)

    assert pipeline._torch_profiler is torch_profiler
    assert pipeline._torch_profile_trace_path == "/tmp/visual-gen-trace-rank-2.json"
    profile.assert_called_once_with(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    )


def test_setup_torch_profiler_requires_profile_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROFILE_TRACE_ENV_VAR_NAME, "/tmp/visual-gen-trace.json")
    pipeline = SimpleNamespace(
        _profile_range=None,
        _torch_profiler=None,
        _torch_profile_trace_path=None,
        rank=0,
    )

    with (
        patch.object(torch.profiler, "profile") as profile,
        patch("tensorrt_llm._torch.visual_gen.pipeline.logger.warning") as warning,
    ):
        BasePipeline._setup_torch_profiler(pipeline)

    profile.assert_not_called()
    warning.assert_called_once()
    assert PROFILE_START_STOP_ENV_VAR_NAME in warning.call_args.args[0]


def test_cuda_profiler_controls_torch_profiler() -> None:
    torch_profiler = MagicMock()
    pipeline = SimpleNamespace(
        _profile_range=(frozenset({0}), frozenset({4})),
        _profiling_active=False,
        _torch_profiler=torch_profiler,
        _torch_profile_trace_path="/tmp/visual-gen-trace-rank-0.json",
        _torch_profile_window=0,
        rank=0,
    )
    pipeline._torch_profile_output_path = MethodType(
        BasePipeline._torch_profile_output_path, pipeline
    )
    cudart = MagicMock()

    with (
        patch("tensorrt_llm._torch.visual_gen.pipeline.torch.cuda.cudart", return_value=cudart),
        patch("tensorrt_llm._torch.visual_gen.pipeline.logger.info"),
    ):
        BasePipeline._cuda_profiler_start(pipeline)
        BasePipeline._cuda_profiler_stop(pipeline)

    cudart.cudaProfilerStart.assert_called_once_with()
    torch_profiler.start.assert_called_once_with()
    torch_profiler.stop.assert_called_once_with()
    torch_profiler.export_chrome_trace.assert_called_once_with("/tmp/visual-gen-trace-rank-0.json")
    cudart.cudaProfilerStop.assert_called_once_with()
    assert not pipeline._profiling_active
    assert pipeline._torch_profiler is None


def test_cuda_profiler_uses_fresh_trace_for_each_window() -> None:
    first_profiler = MagicMock()
    second_profiler = MagicMock()
    pipeline = SimpleNamespace(
        _profile_range=(frozenset({0, 2}), frozenset({1, 3})),
        _profiling_active=False,
        _torch_profiler=first_profiler,
        _torch_profile_trace_path="/tmp/visual-gen-trace-rank-0.json",
        _torch_profile_window=0,
        rank=0,
    )
    pipeline._create_torch_profiler = MethodType(BasePipeline._create_torch_profiler, pipeline)
    pipeline._torch_profile_output_path = MethodType(
        BasePipeline._torch_profile_output_path, pipeline
    )
    cudart = MagicMock()

    with (
        patch.object(torch.profiler, "profile", return_value=second_profiler) as profile,
        patch("tensorrt_llm._torch.visual_gen.pipeline.torch.cuda.cudart", return_value=cudart),
        patch("tensorrt_llm._torch.visual_gen.pipeline.logger.info"),
    ):
        BasePipeline._cuda_profiler_start(pipeline)
        BasePipeline._cuda_profiler_stop(pipeline)
        BasePipeline._cuda_profiler_start(pipeline)
        BasePipeline._cuda_profiler_stop(pipeline)

    profile.assert_called_once()
    first_profiler.export_chrome_trace.assert_called_once_with("/tmp/visual-gen-trace-rank-0.json")
    second_profiler.export_chrome_trace.assert_called_once_with(
        "/tmp/visual-gen-trace-rank-0-window-1.json"
    )
    assert cudart.cudaProfilerStart.call_count == 2
    assert cudart.cudaProfilerStop.call_count == 2
    assert pipeline._torch_profile_window == 2


def test_cuda_profiler_closes_cuda_gate_when_trace_export_fails() -> None:
    torch_profiler = MagicMock()
    torch_profiler.export_chrome_trace.side_effect = RuntimeError("export failed")
    pipeline = SimpleNamespace(
        _profiling_active=True,
        _torch_profiler=torch_profiler,
        _torch_profile_trace_path="/tmp/visual-gen-trace-rank-0.json",
        _torch_profile_window=0,
        rank=0,
    )
    pipeline._torch_profile_output_path = MethodType(
        BasePipeline._torch_profile_output_path, pipeline
    )
    cudart = MagicMock()

    with (
        patch("tensorrt_llm._torch.visual_gen.pipeline.torch.cuda.cudart", return_value=cudart),
        pytest.raises(RuntimeError, match="export failed"),
    ):
        BasePipeline._cuda_profiler_stop(pipeline)

    cudart.cudaProfilerStop.assert_called_once_with()
    assert not pipeline._profiling_active
    assert pipeline._torch_profiler is None


class _RequestProfilingPipeline:
    def __init__(self, profile_range: str) -> None:
        self._profile_range = profile_range
        self._profiling_active = False
        self._is_warmup = False
        self._predenoise_pending = profile_range == "predenoise"
        self._postdenoise_pending = profile_range == "postdenoise"
        self.events: list[str] = []

    def _cuda_profiler_start(self) -> None:
        if not self._profiling_active:
            self.events.append("start")
            self._profiling_active = True

    def _cuda_profiler_stop(self) -> None:
        if self._profiling_active:
            self.events.append("stop")
            self._profiling_active = False

    def infer(self, req: object) -> object:
        self.events.append("text_encode")
        BasePipeline._profile_denoise_start(self)
        self.events.append("denoise")
        BasePipeline._profile_denoise_end(self)
        self.events.append("vae_decode")
        return req


@pytest.mark.parametrize(
    ("profile_range", "expected_events"),
    [
        ("all", ["start", "text_encode", "denoise", "vae_decode", "stop"]),
        ("predenoise", ["start", "text_encode", "stop", "denoise", "vae_decode"]),
        ("postdenoise", ["text_encode", "denoise", "start", "vae_decode", "stop"]),
    ],
)
def test_run_inference_owns_request_profile_boundaries(
    profile_range: str, expected_events: list[str]
) -> None:
    pipeline = _RequestProfilingPipeline(profile_range)
    request = object()

    output = BasePipeline.run_inference(pipeline, request)

    assert output is request
    assert pipeline.events == expected_events


def test_run_inference_defers_predenoise_profile_until_after_warmup() -> None:
    pipeline = _RequestProfilingPipeline("predenoise")
    pipeline._is_warmup = True

    BasePipeline.run_inference(pipeline, object())

    assert pipeline.events == ["text_encode", "denoise", "vae_decode"]
    assert pipeline._predenoise_pending
    assert not pipeline._profiling_active

    pipeline._is_warmup = False
    pipeline.events.clear()
    BasePipeline.run_inference(pipeline, object())

    assert pipeline.events == ["start", "text_encode", "stop", "denoise", "vae_decode"]
    assert not pipeline._predenoise_pending
    assert not pipeline._profiling_active
