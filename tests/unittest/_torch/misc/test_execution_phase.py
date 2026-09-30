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
"""Tests for the process-wide execution phase."""

import pytest

from tensorrt_llm._torch import execution_phase as phase_module
from tensorrt_llm._torch.execution_phase import (
    ExecutionPhase,
    add_execution_phase_listener,
    execution_phase,
    get_execution_phase,
    set_execution_phase,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def isolated_phase_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase_module, "_current_phase", ExecutionPhase.SERVING)
    monkeypatch.setattr(phase_module, "_listeners", [])
    monkeypatch.setattr(phase_module, "_is_capturing_cuda_graph", lambda: False)


def test_initial_phase_is_serving() -> None:
    assert get_execution_phase() is ExecutionPhase.SERVING


def test_transition_notifies_listeners_in_registration_order() -> None:
    seen: list[tuple[str, ExecutionPhase]] = []
    add_execution_phase_listener(lambda phase: seen.append(("first", phase)))
    add_execution_phase_listener(lambda phase: seen.append(("second", phase)))

    set_execution_phase(ExecutionPhase.WARMUP)

    assert get_execution_phase() is ExecutionPhase.WARMUP
    assert seen == [("first", ExecutionPhase.WARMUP), ("second", ExecutionPhase.WARMUP)]


def test_unchanged_phase_does_not_notify() -> None:
    seen: list[ExecutionPhase] = []
    add_execution_phase_listener(seen.append)

    set_execution_phase(ExecutionPhase.SERVING)

    assert seen == []


def test_listener_added_twice_is_notified_once() -> None:
    seen: list[ExecutionPhase] = []
    add_execution_phase_listener(seen.append)
    add_execution_phase_listener(seen.append)

    set_execution_phase(ExecutionPhase.WARMUP)

    assert seen == [ExecutionPhase.WARMUP]


def test_context_manager_restores_the_phase_active_on_entry() -> None:
    set_execution_phase(ExecutionPhase.WARMUP)

    with execution_phase(ExecutionPhase.SERVING):
        assert get_execution_phase() is ExecutionPhase.SERVING
        with execution_phase(ExecutionPhase.WARMUP):
            assert get_execution_phase() is ExecutionPhase.WARMUP
        assert get_execution_phase() is ExecutionPhase.SERVING

    assert get_execution_phase() is ExecutionPhase.WARMUP


def test_context_manager_restores_the_phase_when_the_block_raises() -> None:
    with pytest.raises(KeyError):
        with execution_phase(ExecutionPhase.WARMUP):
            raise KeyError("block failed")

    assert get_execution_phase() is ExecutionPhase.SERVING


def test_transition_during_cuda_graph_capture_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ExecutionPhase] = []
    add_execution_phase_listener(seen.append)
    monkeypatch.setattr(phase_module, "_is_capturing_cuda_graph", lambda: True)

    with pytest.raises(RuntimeError, match="captured"):
        set_execution_phase(ExecutionPhase.WARMUP)

    assert get_execution_phase() is ExecutionPhase.SERVING
    assert seen == []


def test_non_enum_phase_is_rejected() -> None:
    with pytest.raises(TypeError, match="ExecutionPhase"):
        set_execution_phase("warmup")  # type: ignore[arg-type]
