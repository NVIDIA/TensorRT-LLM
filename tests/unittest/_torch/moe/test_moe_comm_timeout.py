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
"""Tests for the phase-aware MoE communication timeout policy and its backend sinks.

The budget and policy tests are CPU-only. The sink tests call each backend's native setter, so
they need the built extensions, and the DeepGEMM and DeepEP round trips also need a GPU. They
import the backends inside the tests, because the CPU-only stage collects this whole file.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

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

SERVING_ENV = "TRTLLM_MOE_COMM_TIMEOUT_SEC"
WARMUP_ENV = "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC"
SERVING_ALIAS = "TRTLLM_MOE_A2A_TIMEOUT_SEC"
WARMUP_ALIAS = "TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC"

_NVLINK_ONE_SIDED_DEFAULT_SEC = 300
_DEEP_EP_DEFAULT_SEC = 100


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


@pytest.mark.cpu_only
class TestBudgetResolution:
    def test_defaults_relax_warmup_and_keep_native_serving(self) -> None:
        assert resolve_moe_comm_timeout_budgets({}) == MoECommTimeoutBudgets(
            warmup_seconds=DEFAULT_WARMUP_TIMEOUT_SEC, serving_seconds=None
        )

    def test_new_names_set_both_budgets(self) -> None:
        budgets = resolve_moe_comm_timeout_budgets({WARMUP_ENV: "900", SERVING_ENV: "120"})

        assert budgets == MoECommTimeoutBudgets(warmup_seconds=900, serving_seconds=120)

    def test_empty_value_counts_as_unset(self) -> None:
        assert resolve_moe_comm_timeout_budgets({SERVING_ENV: " "}).serving_seconds is None

    @pytest.mark.parametrize("raw", ["abc", "1.5", "0", "-3", "86401", "1e3"])
    def test_invalid_value_raises_and_names_the_variable(self, raw: str) -> None:
        with pytest.raises(ValueError, match=SERVING_ENV):
            resolve_moe_comm_timeout_budgets({SERVING_ENV: raw})

    def test_deprecated_aliases_apply_and_warn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        warned: list[str] = []
        monkeypatch.setattr(
            comm_timeout.logger, "warning_once", lambda *msg, key: warned.append(key)
        )

        budgets = resolve_moe_comm_timeout_budgets({WARMUP_ALIAS: "1000", SERVING_ALIAS: "200"})

        assert budgets == MoECommTimeoutBudgets(warmup_seconds=1000, serving_seconds=200)
        assert sorted(warned) == sorted(
            [f"deprecated_env_{WARMUP_ALIAS}", f"deprecated_env_{SERVING_ALIAS}"]
        )

    def test_alias_equal_to_new_name_is_accepted(self) -> None:
        budgets = resolve_moe_comm_timeout_budgets({SERVING_ENV: "200", SERVING_ALIAS: "200"})

        assert budgets.serving_seconds == 200

    def test_alias_conflicting_with_new_name_raises(self) -> None:
        with pytest.raises(ValueError, match=SERVING_ALIAS):
            resolve_moe_comm_timeout_budgets({SERVING_ENV: "100", SERVING_ALIAS: "200"})

    def test_warmup_shorter_than_serving_raises(self) -> None:
        with pytest.raises(ValueError, match=WARMUP_ENV):
            resolve_moe_comm_timeout_budgets({WARMUP_ENV: "100", SERVING_ENV: "200"})

    def test_budgets_select_by_phase(self) -> None:
        budgets = MoECommTimeoutBudgets(warmup_seconds=1800, serving_seconds=None)

        assert budgets.for_phase(ExecutionPhase.WARMUP) == 1800
        assert budgets.for_phase(ExecutionPhase.SERVING) is None


@pytest.mark.cpu_only
class TestPolicy:
    def test_registration_applies_the_current_phase_at_once(self) -> None:
        policy, _ = _make_policy(phase=ExecutionPhase.WARMUP)
        sink = _RecordingSink()

        policy.register(sink)

        assert sink.calls == [DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_transitions_reach_every_sink(self) -> None:
        policy, source = _make_policy(environ={SERVING_ENV: "90"})
        first, second = _RecordingSink("first"), _RecordingSink("second")
        policy.register(first)
        policy.register(second)

        source.transition(ExecutionPhase.WARMUP)
        source.transition(ExecutionPhase.SERVING)

        assert first.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]
        assert second.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]

    def test_serving_without_override_restores_native_defaults(self) -> None:
        policy, source = _make_policy(phase=ExecutionPhase.WARMUP)
        sink = _RecordingSink()
        policy.register(sink)

        source.transition(ExecutionPhase.SERVING)

        assert sink.calls == [DEFAULT_WARMUP_TIMEOUT_SEC, None]

    def test_policy_subscribes_to_phase_changes_once(self) -> None:
        policy, source = _make_policy()

        policy.register(_RecordingSink())
        policy.register(_RecordingSink())

        assert len(source.listeners) == 1

    def test_registering_a_sink_twice_keeps_one_entry(self) -> None:
        policy, source = _make_policy()
        sink = _RecordingSink()
        policy.register(sink)
        policy.register(sink)

        source.transition(ExecutionPhase.WARMUP)

        assert sink.calls == [None, None, DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_unregistered_sink_stops_receiving_transitions(self) -> None:
        policy, source = _make_policy()
        sink = _RecordingSink()
        policy.register(sink)

        policy.unregister(sink)
        source.transition(ExecutionPhase.WARMUP)

        assert sink.calls == [None]

    def test_collected_sink_is_dropped(self) -> None:
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

    def test_sink_whose_setter_fails_is_not_registered(self) -> None:
        policy, source = _make_policy()
        with pytest.raises(RuntimeError, match="setter unavailable"):
            policy.register(_FailingSink())
        healthy = _RecordingSink()
        policy.register(healthy)

        source.transition(ExecutionPhase.WARMUP)

        assert healthy.calls == [None, DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_budgets_resolve_once(self) -> None:
        environ = {SERVING_ENV: "90"}
        policy, source = _make_policy(environ=environ)
        sink = _RecordingSink()
        policy.register(sink)

        environ[SERVING_ENV] = "30"
        source.transition(ExecutionPhase.WARMUP)
        source.transition(ExecutionPhase.SERVING)

        assert sink.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]

    def test_invalid_environment_fails_the_first_registration(self) -> None:
        policy, _ = _make_policy(environ={SERVING_ENV: "never"})

        with pytest.raises(ValueError, match=SERVING_ENV):
            policy.register(_RecordingSink())

    def test_budgets_are_available_before_any_registration(self) -> None:
        environ = {WARMUP_ENV: "900"}
        policy, source = _make_policy(environ=environ)

        first = policy.budgets()
        environ[WARMUP_ENV] = "1200"

        assert first == MoECommTimeoutBudgets(warmup_seconds=900, serving_seconds=None)
        assert policy.budgets() == first
        assert source.listeners == []

    def test_budgets_reject_an_invalid_environment_without_a_sink(self) -> None:
        policy, _ = _make_policy(environ={SERVING_ENV: "soon"})

        with pytest.raises(ValueError, match=SERVING_ENV):
            policy.budgets()

    def test_policy_follows_the_real_execution_phase(self, monkeypatch: pytest.MonkeyPatch) -> None:
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


class TestBackendSinks:
    def test_nvlink_one_sided_sink_sets_the_completion_flag_budget(self) -> None:
        from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided

        sink = nvlink_one_sided._TIMEOUT_SINK
        try:
            sink.set_timeout_seconds(1800)
            assert torch.ops.trtllm.moe_a2a_get_timeout() == 1800
            sink.set_timeout_seconds(None)
            assert torch.ops.trtllm.moe_a2a_get_timeout() == _NVLINK_ONE_SIDED_DEFAULT_SEC
        finally:
            sink.set_timeout_seconds(None)

    @pytest.mark.parametrize("timeout_sec", [-1, 86401])
    def test_moe_a2a_set_timeout_rejects_out_of_range(self, timeout_sec: int) -> None:
        before = torch.ops.trtllm.moe_a2a_get_timeout()

        with pytest.raises(RuntimeError, match="MoE all-to-all timeout"):
            torch.ops.trtllm.moe_a2a_set_timeout(timeout_sec)

        assert torch.ops.trtllm.moe_a2a_get_timeout() == before

    def test_deepgemm_sink_maps_the_native_default_to_zero(self) -> None:
        from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm import (
            _DeepGemmBarrierTimeoutSink,
        )

        calls: list[int] = []
        sink = _DeepGemmBarrierTimeoutSink(
            SimpleNamespace(set_barrier_timeout_seconds=calls.append)
        )

        sink.set_timeout_seconds(1800)
        sink.set_timeout_seconds(None)

        assert calls == [1800, 0]

    def test_deepgemm_sink_sets_the_barrier_timeout(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("DeepGEMM's device runtime needs a GPU")
        dg = pytest.importorskip("tensorrt_llm.deep_gemm")
        from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm import (
            _DeepGemmBarrierTimeoutSink,
        )

        sink = _DeepGemmBarrierTimeoutSink(dg)
        try:
            sink.set_timeout_seconds(1800)
            assert dg.get_barrier_timeout_seconds() == 1800
            sink.set_timeout_seconds(None)
            assert dg.get_barrier_timeout_seconds() == 0
            with pytest.raises(RuntimeError):
                dg.set_barrier_timeout_seconds(86401)
        finally:
            sink.set_timeout_seconds(None)

    def test_deep_ep_sink_sets_the_kernel_timeout(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("DeepEP is built for SM90 and newer")
        from tensorrt_llm._torch.moe.fused_moe import deep_ep_utils

        if not deep_ep_utils.deep_ep_installed:
            pytest.skip("DeepEP is not installed")
        sink = deep_ep_utils._TIMEOUT_SINK
        try:
            sink.set_timeout_seconds(1800)
            assert deep_ep_utils.deep_ep.get_timeout_seconds() == 1800
            sink.set_timeout_seconds(None)
            assert deep_ep_utils.deep_ep.get_timeout_seconds() == _DEEP_EP_DEFAULT_SEC
        finally:
            sink.set_timeout_seconds(None)

    def test_nccl_ep_sink_converts_seconds_to_group_nanoseconds(self) -> None:
        from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import _NcclEpGroupTimeoutSink

        calls: list[int] = []
        sink = _NcclEpGroupTimeoutSink(SimpleNamespace(set_timeout_ns=calls.append))

        sink.set_timeout_seconds(1800)
        sink.set_timeout_seconds(None)

        assert calls == [1_800_000_000_000, 0]
