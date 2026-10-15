# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordering tests for the TRTLLM-Gen FMHA JIT warmup barrier in engine warmup.

The barrier (``torch.ops.trtllm.trtllm_gen_fmha_jit_warmup_drain_and_verify``
plus the rank sync) must run after every sweep has been queued and before
the steady-state MoE all-to-all budget is entered for the graph-shape warmup
and capture passes, and it must run whether or not CUDA graphs are enabled,
so that every attention kernel the warmup derived is compiled before
readiness. These tests drive the runner's capture sequence with the graph
passes, the budget switch and the ops replaced by recorders; the C++ state
machine behind the ops is covered by
``cpp/tests/unit_tests/kernels/fmhaJitWarmupTest.cpp``.
"""

import contextlib
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
    """Records the order of graph passes, barrier calls, rank syncs and budget switches."""

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

    @contextlib.contextmanager
    def steady_state_budget(self):
        self.events.append("budget_enter")
        try:
            yield
        finally:
            self.events.append("budget_exit")


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
        preallocate_padding_dummies=mock.Mock(),
    )
    runner._config = SimpleNamespace(
        attention_backend=SimpleNamespace(
            Metadata=TrtllmAttentionMetadata if trtllm_attention else AttentionMetadata
        )
    )
    runner._warmup_timer = _WarmupTimer(rank=0)
    runner._metrics = {}
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
            runner_module,
            "moe_a2a_steady_state_budget_for_capture",
            side_effect=recorder.steady_state_budget,
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


_CAPTURE_UNDER_BUDGET = ["budget_enter", "warmup", "capture", "budget_exit"]


@pytest.mark.parametrize(
    "cuda_graphs_enabled", [True, False], ids=["graphs_enabled", "graphs_disabled"]
)
def test_barrier_runs_before_the_steady_state_budget(cuda_graphs_enabled: bool) -> None:
    runner = _runner(cuda_graphs_enabled=cuda_graphs_enabled)
    recorder = _Recorder()
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)

    # The graph passes are invoked either way (they return early on their own
    # when graphs are disabled). The barrier precedes the budget switch, so no
    # forward runs under the steady-state all-to-all budget while a compile
    # may still be in flight.
    assert recorder.events == ["drain_and_verify", *_CAPTURE_UNDER_BUDGET]
    assert runner.cuda_graph_runner.is_warmup_only is False
    assert runner.cuda_graph_runner.padding_dummy_requests == {}


def test_barrier_phase_is_not_nested_in_the_capture_phase() -> None:
    # Both phases are recorded, so the warmup summary must not count the
    # drain twice (once on its own and once inside cuda_graph_capture).
    runner = _runner(cuda_graphs_enabled=True)
    recorder = _Recorder()
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    timings = runner._warmup_timer.timings
    assert set(timings) == {"attention_jit_drain_and_verify", "cuda_graph_capture"}
    assert recorder.events.index("drain_and_verify") < recorder.events.index("budget_enter")


def test_barrier_reports_foreground_recovery_of_background_failures() -> None:
    # A background sweep that failed is re-run by the barrier; the op reports
    # how many sweeps it had to verify on the foreground, and warmup proceeds
    # to capture only after that.
    runner = _runner(cuda_graphs_enabled=True)
    recorder = _Recorder(drain_result=2)
    with _patched(recorder), mock.patch.object(runner_module, "logger") as runner_logger:
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["drain_and_verify", *_CAPTURE_UNDER_BUDGET]
    logged = " ".join(str(arg) for call in runner_logger.info.call_args_list for arg in call.args)
    assert "verified 2 background warmup sweep(s)" in logged


def test_barrier_is_skipped_for_non_trtllm_attention_backends() -> None:
    runner = _runner(cuda_graphs_enabled=True, trtllm_attention=False)
    recorder = _Recorder()
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == _CAPTURE_UNDER_BUDGET


def test_barrier_is_skipped_when_async_warmup_is_disabled() -> None:
    # Flag off (the default): the synchronous warmup already compiled
    # everything inline, so there is nothing to drain, verify or sync on.
    runner = _runner(cuda_graphs_enabled=True)
    recorder = _Recorder(async_enabled=False)
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == _CAPTURE_UNDER_BUDGET


def test_ranks_meet_after_the_local_drain_and_before_the_budget() -> None:
    # Each rank only waits for its own compilation; the forwards under the
    # steady-state MoE all-to-all budget would otherwise charge a peer's
    # remaining compile time against that budget, so the ranks meet after
    # draining and before the budget is entered.
    recorder = _Recorder()
    runner = _runner(cuda_graphs_enabled=True, recorder=recorder, multi_rank=True)
    with _patched(recorder):
        runner._run_cuda_graph_warmup_and_capture(resource_manager=None)
    assert recorder.events == ["drain_and_verify", "rank_sync", *_CAPTURE_UNDER_BUDGET]


def test_scheduled_warmup_reaches_the_barrier_before_capture() -> None:
    # Drive the scheduled warmup itself, with every phase that needs a device
    # or a model stubbed out, and check that it arrives at the barrier through
    # the capture helper: the drain happens, and it happens before the budget
    # switch and the graph passes.
    recorder = _Recorder()
    runner = _runner(cuda_graphs_enabled=True)
    runner.mapping = SimpleNamespace(cp_size=1, has_cp_helix=lambda: False)
    runner.guided_decoder = None
    # No context warmups: the attention, general, autotuner and Mamba phases
    # are skipped, which leaves the capture sequence as the only forward.
    runner._context_warmups_supported = False
    stubbed_methods = (
        "_prewarm_engram_kernels",
        "_prewarm_csa2_sparse_indexer",
        "_prewarm_cute_dsl_indexer_q",
        "_ensure_dsa_attn_metadata_for_warmup",
        "_warmup_dg_paged_mqa_logits_metadata",
        "_warmup_cute_dsl_radix_topk",
        "_freeze_eager_workspace_floor",
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(recorder))
        for name in stubbed_methods:
            stack.enter_context(mock.patch.object(DecoderRunner, name, autospec=True))
        stack.enter_context(
            mock.patch.object(
                DecoderRunner,
                "_agree_warmup_flag",
                autospec=True,
                side_effect=lambda self, flag: flag,
            )
        )
        for name in ("warmup_sampling_module", "log_mem_snapshot", "AutoTuner"):
            stack.enter_context(mock.patch.object(runner_module, name))
        runner._warmup_scheduled(resource_manager=None, kv_cache_manager=SimpleNamespace())

    assert recorder.events == ["drain_and_verify", *_CAPTURE_UNDER_BUDGET]
