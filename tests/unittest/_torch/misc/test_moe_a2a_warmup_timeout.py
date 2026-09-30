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
"""The model engine must publish its warmup phase for MoE communication timeouts.

A relaxed warmup budget that latches on into serving would leave hang detection permanently
slow, and CUDA graphs record their launch arguments, so capture must run with serving budgets.
See nvbugs/6482566.
"""

import contextlib
from collections.abc import Iterator

import pytest

from tensorrt_llm._torch import execution_phase as phase_module
from tensorrt_llm._torch.execution_phase import (
    ExecutionPhase,
    get_execution_phase,
    set_execution_phase,
)
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def isolated_phase_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase_module, "_current_phase", ExecutionPhase.SERVING)
    monkeypatch.setattr(phase_module, "_listeners", [])
    monkeypatch.setattr(phase_module, "_is_capturing_cuda_graph", lambda: False)


class _WarmupFlagStub:
    """Borrows the engine's ``is_warmup`` property without building a model.

    The setter only touches ``_is_warmup``, the execution phase, and
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
    """Records the runner mode and execution phase of each CUDA-graph warmup pass."""

    def __init__(self) -> None:
        self.cuda_graph_runner = _GraphRunnerStub()
        self.passes: list[tuple[bool, ExecutionPhase]] = []

    @contextlib.contextmanager
    def maybe_autotune_lora(self) -> Iterator[None]:
        yield

    def _run_cuda_graph_warmup(self, resource_manager: object) -> None:
        self.passes.append((self.cuda_graph_runner.is_warmup_only, get_execution_phase()))


def test_is_warmup_setter_publishes_the_phase_both_ways() -> None:
    stub = _WarmupFlagStub()

    stub.is_warmup = True
    assert get_execution_phase() is ExecutionPhase.WARMUP

    stub.is_warmup = False
    assert get_execution_phase() is ExecutionPhase.SERVING
    assert not stub.is_warmup


def test_only_the_capturing_pass_runs_in_the_serving_phase() -> None:
    engine = _CaptureEngineStub()
    set_execution_phase(ExecutionPhase.WARMUP)

    PyTorchModelEngine._warmup_and_capture_cuda_graphs(engine, resource_manager=None)

    assert engine.passes == [(True, ExecutionPhase.WARMUP), (False, ExecutionPhase.SERVING)]
    assert get_execution_phase() is ExecutionPhase.WARMUP
    assert engine.cuda_graph_runner.padding_dummy_requests == {}
