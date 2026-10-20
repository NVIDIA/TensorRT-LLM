# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model engine must switch the MoE communication timeouts at its warmup transitions.

A relaxed warmup timeout that latches on into serving would leave hang detection permanently
slow, and CUDA graphs record their launch arguments, so capture must use the serving timeouts.
"""

import contextlib
from collections.abc import Iterator

import pytest

from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import (
    DEFAULT_WARMUP_TIMEOUT_SEC,
    MoECommTimeoutGuard,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch) -> MoECommTimeoutGuard:
    guard = MoECommTimeoutGuard(environ={}, is_capturing=lambda: False)
    monkeypatch.setattr(moe_comm_timeout_guard, "_DEFAULT_GUARD", guard)
    return guard


class _RecordingProxy:
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def set_timeout_seconds(self, seconds: int | None) -> None:
        self.calls.append(seconds)


class _WarmupFlagStub:
    """Borrows the engine's ``is_warmup`` property without building a model.

    The setter only touches ``_is_warmup``, the MoE communication timeouts, and
    ``moe_load_balancer_iter_info``, which is a no-op without a load balancer.
    """

    is_warmup = PyTorchModelEngine.is_warmup
    moe_load_balancer_iter_info = PyTorchModelEngine.moe_load_balancer_iter_info
    moe_load_balancer = None


class _GraphRunnerStub:
    def __init__(self) -> None:
        self.is_warmup_only = False
        self.padding_dummy_requests: dict[str, object] = {"stale": object()}

    @contextlib.contextmanager
    def allow_capture(self) -> Iterator[None]:
        yield


class _CaptureGraphsStub(DecoderCudaGraphs):
    """Records the runner mode and the timeout guard state of each CUDA-graph warmup pass."""

    def __init__(self, guard: MoECommTimeoutGuard) -> None:
        self.cuda_graph_runner = _GraphRunnerStub()
        self.passes: list[tuple[bool, bool]] = []
        self._guard = guard

    def run_cuda_graph_warmup(self, resource_manager: object) -> None:
        self.passes.append((self.cuda_graph_runner.is_warmup_only, self._guard.in_warmup))


def test_is_warmup_setter_switches_the_timeouts_both_ways(guard: MoECommTimeoutGuard) -> None:
    proxy = _RecordingProxy()
    guard.register(proxy)
    stub = _WarmupFlagStub()

    stub.is_warmup = True
    assert guard.in_warmup

    stub.is_warmup = False
    assert not guard.in_warmup
    assert not stub.is_warmup
    assert proxy.calls == [None, DEFAULT_WARMUP_TIMEOUT_SEC, None]


def test_only_the_capturing_pass_uses_the_serving_timeouts(guard: MoECommTimeoutGuard) -> None:
    graphs = _CaptureGraphsStub(guard)
    guard.set_warmup(True)

    graphs.warmup_and_capture(resource_manager=None)

    assert graphs.passes == [(True, True), (False, False)]
    assert guard.in_warmup
    assert graphs.cuda_graph_runner.padding_dummy_requests == {}
