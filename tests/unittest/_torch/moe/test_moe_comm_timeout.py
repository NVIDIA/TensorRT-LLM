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
"""Tests for the phase-aware MoE communication timeout policy."""

import gc

import pytest

from tensorrt_llm._torch import execution_phase as phase_module
from tensorrt_llm._torch.execution_phase import (
    ExecutionPhase,
    ExecutionPhaseListener,
    add_execution_phase_listener,
    get_execution_phase,
    set_execution_phase,
)
from tensorrt_llm._torch.moe.fused_moe import comm_timeout
from tensorrt_llm._torch.moe.fused_moe.comm_timeout import (
    DEFAULT_WARMUP_TIMEOUT_SEC,
    MoECommTimeoutBudgets,
    MoECommTimeoutPolicy,
    resolve_moe_comm_timeout_budgets,
)

pytestmark = pytest.mark.cpu_only

SERVING_ENV = "TRTLLM_MOE_COMM_TIMEOUT_SEC"
WARMUP_ENV = "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC"
SERVING_ALIAS = "TRTLLM_MOE_A2A_TIMEOUT_SEC"
WARMUP_ALIAS = "TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC"


class _RecordingSink:
    def __init__(self, name: str = "recording") -> None:
        self.name = name
        self.calls: list[int | None] = []

    def set_timeout_seconds(self, seconds: int | None) -> None:
        self.calls.append(seconds)


class _FailingSink:
    name = "failing"

    def set_timeout_seconds(self, seconds: int | None) -> None:
        raise RuntimeError("setter unavailable")


class _FakePhaseSource:
    def __init__(self, phase: ExecutionPhase) -> None:
        self.phase = phase
        self.listeners: list[ExecutionPhaseListener] = []

    def get(self) -> ExecutionPhase:
        return self.phase

    def add_listener(self, listener: ExecutionPhaseListener) -> None:
        self.listeners.append(listener)

    def transition(self, phase: ExecutionPhase) -> None:
        self.phase = phase
        for listener in self.listeners:
            listener(phase)


def _make_policy(
    environ: dict[str, str] | None = None, phase: ExecutionPhase = ExecutionPhase.SERVING
) -> tuple[MoECommTimeoutPolicy, _FakePhaseSource]:
    source = _FakePhaseSource(phase)
    policy = MoECommTimeoutPolicy(
        environ={} if environ is None else environ,
        get_phase=source.get,
        add_phase_listener=source.add_listener,
    )
    return policy, source


# ---------------------------------------------------------------------------
# Budget resolution
# ---------------------------------------------------------------------------


def test_defaults_relax_warmup_and_keep_native_serving() -> None:
    assert resolve_moe_comm_timeout_budgets({}) == MoECommTimeoutBudgets(
        warmup_seconds=DEFAULT_WARMUP_TIMEOUT_SEC, serving_seconds=None
    )


def test_new_names_set_both_budgets() -> None:
    budgets = resolve_moe_comm_timeout_budgets({WARMUP_ENV: "900", SERVING_ENV: "120"})

    assert budgets == MoECommTimeoutBudgets(warmup_seconds=900, serving_seconds=120)


def test_empty_value_counts_as_unset() -> None:
    assert resolve_moe_comm_timeout_budgets({SERVING_ENV: " "}).serving_seconds is None


@pytest.mark.parametrize("raw", ["abc", "1.5", "0", "-3", "86401", "1e3"])
def test_invalid_value_raises_and_names_the_variable(raw: str) -> None:
    with pytest.raises(ValueError, match=SERVING_ENV):
        resolve_moe_comm_timeout_budgets({SERVING_ENV: raw})


def test_deprecated_aliases_apply_and_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(comm_timeout.logger, "warning_once", lambda *msg, key: warned.append(key))

    budgets = resolve_moe_comm_timeout_budgets({WARMUP_ALIAS: "1000", SERVING_ALIAS: "200"})

    assert budgets == MoECommTimeoutBudgets(warmup_seconds=1000, serving_seconds=200)
    assert sorted(warned) == sorted(
        [f"deprecated_env_{WARMUP_ALIAS}", f"deprecated_env_{SERVING_ALIAS}"]
    )


def test_alias_equal_to_new_name_is_accepted() -> None:
    budgets = resolve_moe_comm_timeout_budgets({SERVING_ENV: "200", SERVING_ALIAS: "200"})

    assert budgets.serving_seconds == 200


def test_alias_conflicting_with_new_name_raises() -> None:
    with pytest.raises(ValueError, match=SERVING_ALIAS):
        resolve_moe_comm_timeout_budgets({SERVING_ENV: "100", SERVING_ALIAS: "200"})


def test_warmup_shorter_than_serving_raises() -> None:
    with pytest.raises(ValueError, match=WARMUP_ENV):
        resolve_moe_comm_timeout_budgets({WARMUP_ENV: "100", SERVING_ENV: "200"})


def test_budgets_select_by_phase() -> None:
    budgets = MoECommTimeoutBudgets(warmup_seconds=1800, serving_seconds=None)

    assert budgets.for_phase(ExecutionPhase.WARMUP) == 1800
    assert budgets.for_phase(ExecutionPhase.SERVING) is None


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_registration_applies_the_current_phase_at_once() -> None:
    policy, _ = _make_policy(phase=ExecutionPhase.WARMUP)
    sink = _RecordingSink()

    policy.register(sink)

    assert sink.calls == [DEFAULT_WARMUP_TIMEOUT_SEC]


def test_transitions_reach_every_sink() -> None:
    policy, source = _make_policy(environ={SERVING_ENV: "90"})
    first, second = _RecordingSink("first"), _RecordingSink("second")
    policy.register(first)
    policy.register(second)

    source.transition(ExecutionPhase.WARMUP)
    source.transition(ExecutionPhase.SERVING)

    assert first.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]
    assert second.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]


def test_serving_without_override_restores_native_defaults() -> None:
    policy, source = _make_policy(phase=ExecutionPhase.WARMUP)
    sink = _RecordingSink()
    policy.register(sink)

    source.transition(ExecutionPhase.SERVING)

    assert sink.calls == [DEFAULT_WARMUP_TIMEOUT_SEC, None]


def test_policy_subscribes_to_phase_changes_once() -> None:
    policy, source = _make_policy()

    policy.register(_RecordingSink())
    policy.register(_RecordingSink())

    assert len(source.listeners) == 1


def test_registering_a_sink_twice_keeps_one_entry() -> None:
    policy, source = _make_policy()
    sink = _RecordingSink()
    policy.register(sink)
    policy.register(sink)

    source.transition(ExecutionPhase.WARMUP)

    assert sink.calls == [None, None, DEFAULT_WARMUP_TIMEOUT_SEC]


def test_unregistered_sink_stops_receiving_transitions() -> None:
    policy, source = _make_policy()
    sink = _RecordingSink()
    policy.register(sink)

    policy.unregister(sink)
    source.transition(ExecutionPhase.WARMUP)

    assert sink.calls == [None]


def test_collected_sink_is_dropped() -> None:
    policy, source = _make_policy()
    calls: list[int | None] = []

    class _TemporarySink:
        name = "temporary"

        def set_timeout_seconds(self, seconds: int | None) -> None:
            calls.append(seconds)

    sink = _TemporarySink()
    policy.register(sink)
    del sink
    gc.collect()

    source.transition(ExecutionPhase.WARMUP)

    assert calls == [None]


def test_sink_whose_setter_fails_is_not_registered() -> None:
    policy, source = _make_policy()
    with pytest.raises(RuntimeError, match="setter unavailable"):
        policy.register(_FailingSink())
    healthy = _RecordingSink()
    policy.register(healthy)

    source.transition(ExecutionPhase.WARMUP)

    assert healthy.calls == [None, DEFAULT_WARMUP_TIMEOUT_SEC]


def test_budgets_resolve_once() -> None:
    environ = {SERVING_ENV: "90"}
    policy, source = _make_policy(environ=environ)
    sink = _RecordingSink()
    policy.register(sink)

    environ[SERVING_ENV] = "30"
    source.transition(ExecutionPhase.WARMUP)
    source.transition(ExecutionPhase.SERVING)

    assert sink.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]


def test_invalid_environment_fails_the_first_registration() -> None:
    policy, _ = _make_policy(environ={SERVING_ENV: "never"})

    with pytest.raises(ValueError, match=SERVING_ENV):
        policy.register(_RecordingSink())


def test_budgets_are_available_before_any_registration() -> None:
    environ = {WARMUP_ENV: "900"}
    policy, source = _make_policy(environ=environ)

    first = policy.budgets()
    environ[WARMUP_ENV] = "1200"

    assert first == MoECommTimeoutBudgets(warmup_seconds=900, serving_seconds=None)
    assert policy.budgets() == first
    assert source.listeners == []


def test_budgets_reject_an_invalid_environment_without_a_sink() -> None:
    policy, _ = _make_policy(environ={SERVING_ENV: "soon"})

    with pytest.raises(ValueError, match=SERVING_ENV):
        policy.budgets()


def test_policy_follows_the_real_execution_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase_module, "_current_phase", ExecutionPhase.SERVING)
    monkeypatch.setattr(phase_module, "_listeners", [])
    monkeypatch.setattr(phase_module, "_is_capturing_cuda_graph", lambda: False)
    policy = MoECommTimeoutPolicy(
        environ={},
        get_phase=get_execution_phase,
        add_phase_listener=add_execution_phase_listener,
    )
    sink = _RecordingSink()
    policy.register(sink)

    set_execution_phase(ExecutionPhase.WARMUP)
    set_execution_phase(ExecutionPhase.SERVING)

    assert sink.calls == [None, DEFAULT_WARMUP_TIMEOUT_SEC, None]
