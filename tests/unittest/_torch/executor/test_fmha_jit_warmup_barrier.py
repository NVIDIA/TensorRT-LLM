# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordering tests for the TRTLLM-Gen FMHA JIT warmup barrier in engine warmup.

The barrier (``torch.ops.trtllm.trtllm_gen_fmha_jit_warmup_drain_and_verify``)
must run after the graph-shape warmup pass and before any CUDA graph is
captured, and it must run whether or not CUDA graphs are enabled, so that
every attention kernel the warmup derived is compiled before readiness. These
tests drive the runner's capture sequence with the graph passes and the ops
replaced by recorders; the C++ state machine behind the ops is covered by
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
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as runner_module
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer

pytestmark = pytest.mark.cpu_only


class _Recorder:
    """Records the order of graph passes, barrier calls and rank syncs."""

    def __init__(self, drain_result: int = 0, async_enabled: bool = True):
        self.events: list[str] = []
        self.drain_result = drain_result
        self.async_enabled = async_enabled

    def run_cuda_graph_warmup(self, runner, resource_manager):
        self.events.append("warmup" if runner.cuda_graph_runner.is_warmup_only else "capture")

    def drain_and_verify(self):
        self.events.append("drain_and_verify")
        return self.drain_result

    def allgather(self, value):
        self.events.append("rank_sync")
        return [value, value]


def _runner(
    *,
    cuda_graphs_enabled: bool,
    trtllm_attention: bool = True,
    recorder: _Recorder | None = None,
    multi_rank: bool = False,
) -> DecoderRunner:
    """Build a runner carrying only what the capture sequence reads."""
    runner = object.__new__(DecoderRunner)

    @contextlib.contextmanager
    def allow_capture():
        yield

    runner.cuda_graph_runner = SimpleNamespace(
        enabled=cuda_graphs_enabled,
        is_warmup_only=False,
        padding_dummy_requests={"stale": object()},
        allow_capture=allow_capture,
    )
    runner._config = SimpleNamespace(
        attention_backend=SimpleNamespace(
            Metadata=TrtllmAttentionMetadata if trtllm_attention else AttentionMetadata
        )
    )
    runner._warmup_timer = _WarmupTimer(rank=0)
    # The forward-group agreement primitive the barrier synchronizes on; None
    # is what the real helper returns for a single rank (and under DWDP).
    runner._warmup_agreement_allgather = (
        lambda: recorder.allgather if multi_rank and recorder else None
    )
    return runner


@contextlib.contextmanager
def _patched(recorder: _Recorder):
    with (
        mock.patch.object(
            DecoderRunner,
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
        mock.patch.object(
            torch.ops.trtllm,
            "trtllm_gen_fmha_async_jit_warmup_enabled",
            create=True,
            side_effect=lambda: recorder.async_enabled,
        ),
    ):
        yield


@pytest.mark.parametrize(
    "cuda_graphs_enabled", [True, False], ids=["graphs_enabled", "graphs_disabled"]
)
def test_barrier_runs_between_graph_warmup_and_capture(cuda_graphs_enabled: bool) -> None:
    runner = _runner(cuda_graphs_enabled=cuda_graphs_enabled)
    recorder = _Recorder()
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)

    # The graph passes are invoked either way (they return early on their own
    # when graphs are disabled); the barrier sits strictly between them, so
    # verification finishes before the first captured launch.
    assert recorder.events == ["warmup", "drain_and_verify", "capture"]
    assert runner.cuda_graph_runner.is_warmup_only is False
    assert runner.cuda_graph_runner.padding_dummy_requests == {}


def test_barrier_does_not_depend_on_general_warmup() -> None:
    # The capture sequence runs the barrier unconditionally: it never consults
    # the general-warmup gate (``can_run_general_warmup`` is decided earlier in
    # the scheduled warmup and only scopes the general / memory-pool phases).
    runner = _runner(cuda_graphs_enabled=False)
    recorder = _Recorder()
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "drain_and_verify", "capture"]
    source = inspect.getsource(DecoderRunner._run_cuda_graph_warmup_and_capture)
    assert "can_run_general_warmup" not in source


def test_barrier_reports_foreground_recovery_of_background_failures() -> None:
    # A background sweep that failed is re-run by the barrier; the op reports
    # how many sweeps it had to verify on the foreground, and warmup proceeds
    # to capture only after that.
    runner = _runner(cuda_graphs_enabled=True)
    recorder = _Recorder(drain_result=2)
    with _patched(recorder), mock.patch.object(runner_module, "logger") as runner_logger:
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "drain_and_verify", "capture"]
    logged = " ".join(str(arg) for call in runner_logger.info.call_args_list for arg in call.args)
    assert "verified 2 background warmup sweep(s)" in logged


def test_barrier_is_skipped_for_non_trtllm_attention_backends() -> None:
    runner = _runner(cuda_graphs_enabled=True, trtllm_attention=False)
    recorder = _Recorder()
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "capture"]


def test_barrier_is_skipped_when_async_warmup_is_disabled() -> None:
    # Flag off (the default): the synchronous warmup already compiled
    # everything inline, so there is nothing to drain, verify or sync on.
    runner = _runner(cuda_graphs_enabled=True)
    recorder = _Recorder(async_enabled=False)
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "capture"]


def test_ranks_meet_after_the_local_drain_and_before_capture() -> None:
    # Each rank only waits for its own compilation; the capture forwards run
    # collectives under the steady-state MoE all-to-all budget, so the ranks
    # must meet after draining and before the first captured forward.
    recorder = _Recorder()
    runner = _runner(cuda_graphs_enabled=True, recorder=recorder, multi_rank=True)
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["warmup", "drain_and_verify", "rank_sync", "capture"]


def test_warmup_routes_capture_through_the_helper() -> None:
    # The scheduled warmup must not call ``_run_cuda_graph_warmup`` directly:
    # the only path to capture is the helper that carries the barrier, so the
    # barrier cannot be reordered or dropped by a change to the warmup flow
    # alone. ``warmup`` delegates to ``_warmup_scheduled``, where the capture
    # call lives.
    scheduled = inspect.getsource(inspect.unwrap(DecoderRunner._warmup_scheduled))
    assert "self._run_cuda_graph_warmup_and_capture(" in scheduled
    for method in (DecoderRunner.warmup, DecoderRunner._warmup_scheduled):
        assert "self._run_cuda_graph_warmup(" not in inspect.getsource(inspect.unwrap(method))
