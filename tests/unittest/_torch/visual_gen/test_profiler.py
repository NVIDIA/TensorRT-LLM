# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
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
    torch_profiler = MagicMock()

    with patch.object(torch.profiler, "profile", return_value=torch_profiler) as profile:
        BasePipeline._setup_torch_profiler(pipeline)

    assert pipeline._torch_profiler is torch_profiler
    assert pipeline._torch_profile_trace_path == "/tmp/visual-gen-trace-rank-2.json"
    profile.assert_called_once_with(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.XPU,
        ],
        record_shapes=True,
        with_modules=True,
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
        rank=0,
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
