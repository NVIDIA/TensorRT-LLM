# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordering tests for the TRTLLM-Gen FMHA JIT warmup barrier in decoder warmup.

The barrier (``torch.ops.trtllm.trtllm_gen_fmha_jit_warmup_drain_and_verify``
plus the rank sync) must run after every sweep has been queued and before
the graph-shape warmup and capture passes, so that no compile (or wait on a
background compile) lands inside a forward that runs MoE all-to-alls, and so
that every attention kernel the warmup derived is compiled before readiness.
These tests drive ``DecoderWarmup.run`` with the graph passes, the MoE
serving-timeout switch and the ops replaced by recorders; the C++ state
machine behind the ops is covered by
``cpp/tests/unit_tests/kernels/fmhaJitWarmupTest.cpp``.
"""

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as graphs_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import warmup as warmup_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer

pytestmark = pytest.mark.cpu_only


class _Recorder:
    """Records the order of barrier calls, rank syncs, timeout switches and graph passes."""

    def __init__(self, drain_result: int = 0, async_enabled: bool = True):
        self.events: list[str] = []
        self.drain_result = drain_result
        self.async_enabled = async_enabled

    def drain_and_verify(self) -> int:
        self.events.append("drain_and_verify")
        return self.drain_result

    def allgather(self, value):
        self.events.append("rank_sync")
        return [value, value]

    @contextlib.contextmanager
    def serving_timeouts(self) -> Iterator[None]:
        self.events.append("serving_timeouts_enter")
        try:
            yield
        finally:
            self.events.append("serving_timeouts_exit")


class _GraphRunnerStub:
    def __init__(self) -> None:
        self.is_warmup_only = False
        self.padding_dummy_requests: dict[str, object] = {"stale": object()}

    @contextlib.contextmanager
    def allow_capture(self) -> Iterator[None]:
        yield

    def preallocate_padding_dummies(self, resource_manager: object) -> None:
        pass


class _RecordingGraphs(DecoderCudaGraphs):
    """The real two-pass ``warmup_and_capture`` over a recording graph pass."""

    def __init__(self, recorder: _Recorder) -> None:
        self.cuda_graph_runner = _GraphRunnerStub()
        self._recorder = recorder

    def run_cuda_graph_warmup(self, resource_manager: object) -> None:
        self._recorder.events.append(
            "graph_warmup" if self.cuda_graph_runner.is_warmup_only else "graph_capture"
        )


def _engine(
    recorder: _Recorder, *, trtllm_attention: bool = True, world_size: int = 1
) -> DecoderWarmup:
    """Build a warmup engine carrying only what the scheduled path reads.

    The context warmups are off, which skips the attention, general, autotuner
    and Mamba phases, so the graph-shape warmup and capture passes are the only
    forwards. ``world_size > 1`` gives the forward group a recording allgather.
    """
    dist = (
        SimpleNamespace(world_size=world_size, allgather=recorder.allgather)
        if world_size > 1
        else None
    )
    mapping = SimpleNamespace(cp_size=1, tp_size=1, dwdp_enabled=False, has_cp_helix=lambda: False)
    engine = object.__new__(DecoderWarmup)
    engine._agreement = WarmupAgreement(dist, mapping)
    engine.dist = dist
    engine.mapping = mapping
    engine._config = SimpleNamespace(
        attention_backend=SimpleNamespace(
            Metadata=TrtllmAttentionMetadata if trtllm_attention else AttentionMetadata
        )
    )
    engine._warmup_timer = _WarmupTimer(rank=0)
    engine._metrics = {}
    engine._graphs = _RecordingGraphs(recorder)
    engine._inputs = SimpleNamespace(guided_decoder=None)
    engine._metadata = SimpleNamespace(attn_metadata=None)
    engine._context_warmups_supported = False
    engine._eager_workspace_reclaimable = False
    engine.model = SimpleNamespace(modules=lambda: iter(()))
    # Model-specific prewarms the scheduled path runs unconditionally; each
    # needs a model or a device to decide it has nothing to do.
    engine._prewarm_engram_kernels = mock.Mock()
    engine._prewarm_csa2_sparse_indexer = mock.Mock()
    engine._prewarm_minimax_m3_msa_indexer = mock.Mock()
    engine._prewarm_cute_dsl_indexer_q = mock.Mock()
    engine._ensure_dsa_attn_metadata_for_warmup = mock.Mock()
    return engine


def _resource_manager() -> SimpleNamespace:
    kv_cache_manager = SimpleNamespace()
    return SimpleNamespace(get_resource_manager=lambda key: kv_cache_manager)


@contextlib.contextmanager
def _patched(recorder: _Recorder) -> Iterator[None]:
    with (
        mock.patch.object(warmup_module, "warmup_sampling_module"),
        mock.patch.object(warmup_module, "warmup_named_allreduce_topologies"),
        mock.patch.object(warmup_module, "log_mem_snapshot"),
        mock.patch.object(warmup_module, "AutoTuner"),
        mock.patch.object(
            graphs_module, "moe_comm_serving_timeouts", side_effect=recorder.serving_timeouts
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


# The graph-shape warmup pass keeps the warmup MoE timeouts; only the capture
# pass switches to the serving ones.
_GRAPH_PASSES = ["graph_warmup", "serving_timeouts_enter", "graph_capture", "serving_timeouts_exit"]


def test_barrier_runs_before_the_graph_shape_warmup_and_capture() -> None:
    recorder = _Recorder()
    engine = _engine(recorder)
    with _patched(recorder):
        engine.run(_resource_manager())

    assert recorder.events == ["drain_and_verify", *_GRAPH_PASSES]
    assert engine._graphs.cuda_graph_runner.is_warmup_only is False
    assert engine._graphs.cuda_graph_runner.padding_dummy_requests == {}


def test_barrier_phase_is_not_nested_in_the_capture_phase() -> None:
    # Both phases are recorded, so the warmup summary must not count the
    # drain twice (once on its own and once inside cuda_graph_capture).
    recorder = _Recorder()
    engine = _engine(recorder)
    with _patched(recorder):
        engine.run(_resource_manager())
    timings = engine._warmup_timer.timings
    assert {"attention_jit_drain_and_verify", "cuda_graph_capture"} <= set(timings)
    assert recorder.events.index("drain_and_verify") < recorder.events.index("graph_warmup")


def test_barrier_reports_foreground_recovery_of_background_failures() -> None:
    # A background sweep that failed is re-run by the barrier; the op reports
    # how many sweeps it had to verify on the foreground, and warmup proceeds
    # to the graph passes only after that.
    recorder = _Recorder(drain_result=2)
    engine = _engine(recorder)
    with _patched(recorder), mock.patch.object(warmup_module, "logger") as warmup_logger:
        engine.run(_resource_manager())
    assert recorder.events == ["drain_and_verify", *_GRAPH_PASSES]
    logged = " ".join(str(arg) for call in warmup_logger.info.call_args_list for arg in call.args)
    assert "verified 2 background warmup sweep(s)" in logged


def test_barrier_is_skipped_for_non_trtllm_attention_backends() -> None:
    recorder = _Recorder()
    engine = _engine(recorder, trtllm_attention=False)
    with _patched(recorder):
        engine.run(_resource_manager())
    assert recorder.events == _GRAPH_PASSES


def test_barrier_is_skipped_when_async_warmup_is_disabled() -> None:
    # Flag off (the default): the synchronous warmup already compiled
    # everything inline, so there is nothing to drain, verify or sync on.
    recorder = _Recorder(async_enabled=False)
    engine = _engine(recorder)
    with _patched(recorder):
        engine.run(_resource_manager())
    assert recorder.events == _GRAPH_PASSES


def test_ranks_meet_after_the_local_drain_and_before_the_graph_passes() -> None:
    # Each rank only waits for its own compilation; a rank entering the graph
    # passes while a peer is still compiling would wait on that peer inside an
    # all-to-all, so the ranks meet after draining and before the first pass.
    recorder = _Recorder()
    engine = _engine(recorder, world_size=2)
    with _patched(recorder):
        engine.run(_resource_manager())
    # The leading sync is the phase-entry agreement that precedes every
    # warmup phase; the one after the drain is the barrier's.
    assert recorder.events == ["rank_sync", "drain_and_verify", "rank_sync", *_GRAPH_PASSES]
