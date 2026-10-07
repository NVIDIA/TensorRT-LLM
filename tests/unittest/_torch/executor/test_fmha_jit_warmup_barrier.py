# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordering tests for the TRTLLM-Gen FMHA JIT warmup barrier in engine warmup.

The barrier (``torch.ops.trtllm.trtllm_gen_fmha_jit_warmup_drain_and_verify``)
must run after the graph-shape warmup pass and before any CUDA graph is
captured, and it must run whether or not CUDA graphs are enabled, so that
every attention kernel the warmup derived is compiled before readiness. These
tests drive the engine's capture sequence with the graph passes and the op
replaced by recorders; the C++ state machine behind the op is covered by
``cpp/tests/unit_tests/kernels/fmhaJitWarmupTest.cpp``.
"""

import contextlib
import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor import model_engine as model_engine_module
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer

pytestmark = pytest.mark.cpu_only


class _Recorder:
    """Records the order of graph passes and barrier calls."""

    def __init__(self, drain_result: int = 0):
        self.events: list[str] = []
        self.drain_result = drain_result

    def run_cuda_graph_warmup(self, engine, resource_manager):
        self.events.append(
            "warmup" if engine.cuda_graph_runner.is_warmup_only else "capture")

    def drain_and_verify(self):
        self.events.append("drain_and_verify")
        return self.drain_result


def _engine(*, cuda_graphs_enabled: bool,
            trtllm_attention: bool = True) -> PyTorchModelEngine:
    """Build an engine carrying only what the capture sequence reads."""
    engine = object.__new__(PyTorchModelEngine)

    @contextlib.contextmanager
    def allow_capture():
        yield

    engine.cuda_graph_runner = SimpleNamespace(
        enabled=cuda_graphs_enabled,
        is_warmup_only=False,
        padding_dummy_requests={"stale": object()},
        allow_capture=allow_capture,
    )
    engine.attn_backend = SimpleNamespace(
        Metadata=TrtllmAttentionMetadata if trtllm_attention else
        AttentionMetadata)
    engine._warmup_timer = _WarmupTimer(rank=0)
    return engine


@contextlib.contextmanager
def _patched(engine: PyTorchModelEngine, recorder: _Recorder):
    with (
            mock.patch.object(
                PyTorchModelEngine,
                "_run_cuda_graph_warmup",
                autospec=True,
                side_effect=recorder.run_cuda_graph_warmup,
            ),
            mock.patch.object(
                torch.ops.trtllm,
                "trtllm_gen_fmha_jit_warmup_drain_and_verify",
                create=True,
                side_effect=recorder.drain_and_verify,
            ),
    ):
        yield


@pytest.mark.parametrize("cuda_graphs_enabled", [True, False],
                         ids=["graphs_enabled", "graphs_disabled"])
def test_barrier_runs_between_graph_warmup_and_capture(
        cuda_graphs_enabled: bool) -> None:
    engine = _engine(cuda_graphs_enabled=cuda_graphs_enabled)
    recorder = _Recorder()
    with _patched(engine, recorder):
        engine._run_cuda_graph_warmup_and_capture(resource_manager=None)

    # The graph passes are invoked either way (they return early on their own
    # when graphs are disabled); the barrier sits strictly between them, so
    # verification finishes before the first captured launch.
    assert recorder.events == ["warmup", "drain_and_verify", "capture"]
    assert engine.cuda_graph_runner.is_warmup_only is False
    assert engine.cuda_graph_runner.padding_dummy_requests == {}


def test_barrier_does_not_depend_on_general_warmup() -> None:
    # The capture sequence runs the barrier unconditionally: it never consults
    # the general-warmup gate (``can_run_general_warmup`` is decided earlier in
    # ``warmup`` and only scopes the general / memory-pool phases).
    engine = _engine(cuda_graphs_enabled=False)
    recorder = _Recorder()
    with _patched(engine, recorder):
        engine._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "drain_and_verify", "capture"]
    source = inspect.getsource(
        PyTorchModelEngine._run_cuda_graph_warmup_and_capture)
    assert "can_run_general_warmup" not in source


def test_barrier_reports_foreground_recovery_of_background_failures() -> None:
    # A background sweep that failed is re-run by the barrier; the op reports
    # how many sweeps it had to verify on the foreground, and warmup proceeds
    # to capture only after that.
    engine = _engine(cuda_graphs_enabled=True)
    recorder = _Recorder(drain_result=2)
    with _patched(engine, recorder), \
            mock.patch.object(model_engine_module, "logger") as engine_logger:
        engine._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "drain_and_verify", "capture"]
    logged = " ".join(
        str(arg) for call in engine_logger.info.call_args_list
        for arg in call.args)
    assert "verified 2 background warmup sweep(s)" in logged


def test_barrier_is_skipped_for_non_trtllm_attention_backends() -> None:
    engine = _engine(cuda_graphs_enabled=True, trtllm_attention=False)
    recorder = _Recorder()
    with _patched(engine, recorder):
        engine._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "capture"]


def test_warmup_routes_capture_through_the_helper() -> None:
    # ``warmup`` must not call ``_run_cuda_graph_warmup`` directly: the only
    # path to capture is the helper that carries the barrier, so the barrier
    # cannot be reordered or dropped by a change to ``warmup`` alone.
    source = inspect.getsource(PyTorchModelEngine.warmup)
    assert "self._run_cuda_graph_warmup_and_capture(" in source
    assert "self._run_cuda_graph_warmup(" not in source
