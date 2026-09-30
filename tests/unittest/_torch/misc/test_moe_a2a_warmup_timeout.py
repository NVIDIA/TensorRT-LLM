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
"""The model engine must switch the MoE communication timeouts at its warmup transitions.

A relaxed warmup timeout that latches on into serving would leave hang detection permanently
slow, and CUDA graphs record their launch arguments, so capture must use the serving timeouts.
See nvbugs/6482566.
"""

import contextlib
from collections.abc import Iterator

import pytest

from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import (
    DEFAULT_WARMUP_TIMEOUT_SEC,
    MoECommTimeoutGuard,
)
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


class _CaptureEngineStub:
    """Records the runner mode and the timeout guard state of each CUDA-graph warmup pass."""

    def __init__(self, guard: MoECommTimeoutGuard) -> None:
        self.cuda_graph_runner = _GraphRunnerStub()
        self.passes: list[tuple[bool, bool]] = []
        self._guard = guard

    @contextlib.contextmanager
    def maybe_autotune_lora(self) -> Iterator[None]:
        yield

    def _run_cuda_graph_warmup(self, resource_manager: object) -> None:
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
    engine = _CaptureEngineStub(guard)
    guard.set_warmup(True)

    PyTorchModelEngine._warmup_and_capture_cuda_graphs(engine, resource_manager=None)

    assert engine.passes == [(True, True), (False, False)]
    assert guard.in_warmup
    assert engine.cuda_graph_runner.padding_dummy_requests == {}
