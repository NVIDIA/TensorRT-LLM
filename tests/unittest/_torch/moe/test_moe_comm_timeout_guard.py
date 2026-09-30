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
"""Tests for the MoE communication timeout guard and its backend proxies.

The budget and guard tests are CPU-only. The proxy tests call each backend's native setter, so
they need the built extensions, and the DeepGEMM and DeepEP round trips also need a GPU. They
import the backends inside the tests, because the CPU-only stage collects this whole file.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import (
    DEFAULT_WARMUP_TIMEOUT_SEC,
    MoECommTimeoutBudgets,
    MoECommTimeoutGuard,
    moe_comm_serving_timeouts,
    register_moe_comm_timeout_proxy,
    resolve_moe_comm_timeout_budgets,
    set_moe_comm_warmup,
)

SERVING_ENV = "TRTLLM_MOE_COMM_TIMEOUT_SEC"
WARMUP_ENV = "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC"
SERVING_ALIAS = "TRTLLM_MOE_A2A_TIMEOUT_SEC"
WARMUP_ALIAS = "TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC"

_NVLINK_ONE_SIDED_DEFAULT_SEC = 300
_DEEP_EP_DEFAULT_SEC = 100


class _RecordingProxy:
    def __init__(self, name: str = "recording") -> None:
        self.name = name
        self.calls: list[int | None] = []

    def set_timeout_seconds(self, seconds: int | None) -> None:
        self.calls.append(seconds)


class _FailingProxy:
    name = "failing"

    def set_timeout_seconds(self, seconds: int | None) -> None:
        raise RuntimeError("setter unavailable")


class _CaptureState:
    def __init__(self) -> None:
        self.active = False

    def __call__(self) -> bool:
        return self.active


def _make_guard(
    environ: dict[str, str] | None = None, capture: _CaptureState | None = None
) -> MoECommTimeoutGuard:
    return MoECommTimeoutGuard(
        environ={} if environ is None else environ,
        is_capturing=_CaptureState() if capture is None else capture,
    )


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
            moe_comm_timeout_guard.logger, "warning_once", lambda *msg, key: warned.append(key)
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

    def test_budgets_select_warmup_or_serving(self) -> None:
        budgets = MoECommTimeoutBudgets(warmup_seconds=1800, serving_seconds=None)

        assert budgets.select(in_warmup=True) == 1800
        assert budgets.select(in_warmup=False) is None


@pytest.mark.cpu_only
class TestGuard:
    def test_guard_starts_in_serving(self) -> None:
        guard = _make_guard(environ={SERVING_ENV: "90"})
        proxy = _RecordingProxy()

        guard.register(proxy)

        assert not guard.in_warmup
        assert proxy.calls == [90]

    def test_registration_applies_the_current_timeout_at_once(self) -> None:
        guard = _make_guard()
        guard.set_warmup(True)
        proxy = _RecordingProxy()

        guard.register(proxy)

        assert proxy.calls == [DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_transitions_reach_every_proxy(self) -> None:
        guard = _make_guard(environ={SERVING_ENV: "90"})
        first, second = _RecordingProxy("first"), _RecordingProxy("second")
        guard.register(first)
        guard.register(second)

        guard.set_warmup(True)
        guard.set_warmup(False)

        assert first.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]
        assert second.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]

    def test_serving_without_override_restores_native_defaults(self) -> None:
        guard = _make_guard()
        guard.set_warmup(True)
        proxy = _RecordingProxy()
        guard.register(proxy)

        guard.set_warmup(False)

        assert proxy.calls == [DEFAULT_WARMUP_TIMEOUT_SEC, None]

    def test_repeating_the_current_state_pushes_nothing(self) -> None:
        guard = _make_guard()
        proxy = _RecordingProxy()
        guard.register(proxy)

        guard.set_warmup(False)
        guard.set_warmup(True)
        guard.set_warmup(True)

        assert proxy.calls == [None, DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_registering_a_proxy_twice_keeps_one_entry(self) -> None:
        guard = _make_guard()
        proxy = _RecordingProxy()
        guard.register(proxy)
        guard.register(proxy)

        guard.set_warmup(True)

        assert proxy.calls == [None, None, DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_unregistered_proxy_stops_receiving_transitions(self) -> None:
        guard = _make_guard()
        proxy = _RecordingProxy()
        guard.register(proxy)

        guard.unregister(proxy)
        guard.set_warmup(True)

        assert proxy.calls == [None]

    def test_collected_proxy_is_dropped(self) -> None:
        guard = _make_guard()
        calls: list[int | None] = []

        class _TemporaryProxy:
            name = "temporary"

            def set_timeout_seconds(self, seconds: int | None) -> None:
                calls.append(seconds)

        proxy = _TemporaryProxy()
        guard.register(proxy)
        del proxy
        gc.collect()

        guard.set_warmup(True)

        assert calls == [None]

    def test_proxy_whose_setter_fails_is_not_registered(self) -> None:
        guard = _make_guard()
        with pytest.raises(RuntimeError, match="setter unavailable"):
            guard.register(_FailingProxy())
        healthy = _RecordingProxy()
        guard.register(healthy)

        guard.set_warmup(True)

        assert healthy.calls == [None, DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_budgets_resolve_once(self) -> None:
        environ = {SERVING_ENV: "90"}
        guard = _make_guard(environ=environ)
        proxy = _RecordingProxy()
        guard.register(proxy)

        environ[SERVING_ENV] = "30"
        guard.set_warmup(True)
        guard.set_warmup(False)

        assert proxy.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90]

    def test_invalid_environment_fails_the_first_registration(self) -> None:
        guard = _make_guard(environ={SERVING_ENV: "never"})

        with pytest.raises(ValueError, match=SERVING_ENV):
            guard.register(_RecordingProxy())

    def test_transitions_without_proxies_leave_the_environment_unread(self) -> None:
        guard = _make_guard(environ={SERVING_ENV: "never"})

        guard.set_warmup(True)
        guard.set_warmup(False)

        with pytest.raises(ValueError, match=SERVING_ENV):
            guard.budgets()

    def test_budgets_are_available_before_any_registration(self) -> None:
        environ = {WARMUP_ENV: "900"}
        guard = _make_guard(environ=environ)

        first = guard.budgets()
        environ[WARMUP_ENV] = "1200"

        assert first == MoECommTimeoutBudgets(warmup_seconds=900, serving_seconds=None)
        assert guard.budgets() == first

    def test_switching_during_cuda_graph_capture_raises(self) -> None:
        capture = _CaptureState()
        guard = _make_guard(capture=capture)
        proxy = _RecordingProxy()
        guard.register(proxy)
        capture.active = True

        with pytest.raises(RuntimeError, match="captured"):
            guard.set_warmup(True)

        assert not guard.in_warmup
        assert proxy.calls == [None]

    def test_serving_timeouts_apply_serving_and_restore_warmup(self) -> None:
        guard = _make_guard(environ={SERVING_ENV: "90"})
        guard.set_warmup(True)
        proxy = _RecordingProxy()
        guard.register(proxy)

        with guard.serving_timeouts():
            assert not guard.in_warmup
            assert proxy.calls == [DEFAULT_WARMUP_TIMEOUT_SEC, 90]

        assert guard.in_warmup
        assert proxy.calls == [DEFAULT_WARMUP_TIMEOUT_SEC, 90, DEFAULT_WARMUP_TIMEOUT_SEC]

    def test_serving_timeouts_restore_when_the_block_raises(self) -> None:
        guard = _make_guard()
        guard.set_warmup(True)

        with pytest.raises(KeyError), guard.serving_timeouts():
            raise KeyError("boom")

        assert guard.in_warmup

    def test_module_functions_use_the_process_wide_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        guard = _make_guard(environ={SERVING_ENV: "90"})
        monkeypatch.setattr(moe_comm_timeout_guard, "_DEFAULT_GUARD", guard)
        proxy = _RecordingProxy()

        register_moe_comm_timeout_proxy(proxy)
        set_moe_comm_warmup(True)
        with moe_comm_serving_timeouts():
            pass

        assert proxy.calls == [90, DEFAULT_WARMUP_TIMEOUT_SEC, 90, DEFAULT_WARMUP_TIMEOUT_SEC]


class TestBackendProxies:
    def test_nvlink_one_sided_proxy_sets_the_completion_flag_budget(self) -> None:
        from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided

        proxy = nvlink_one_sided._TIMEOUT_PROXY
        try:
            proxy.set_timeout_seconds(1800)
            assert torch.ops.trtllm.moe_a2a_get_timeout() == 1800
            proxy.set_timeout_seconds(None)
            assert torch.ops.trtllm.moe_a2a_get_timeout() == _NVLINK_ONE_SIDED_DEFAULT_SEC
        finally:
            proxy.set_timeout_seconds(None)

    @pytest.mark.parametrize("timeout_sec", [-1, 86401])
    def test_moe_a2a_set_timeout_rejects_out_of_range(self, timeout_sec: int) -> None:
        before = torch.ops.trtllm.moe_a2a_get_timeout()

        with pytest.raises(RuntimeError, match="MoE all-to-all timeout"):
            torch.ops.trtllm.moe_a2a_set_timeout(timeout_sec)

        assert torch.ops.trtllm.moe_a2a_get_timeout() == before

    def test_deepgemm_proxy_maps_the_native_default_to_zero(self) -> None:
        from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm import (
            _DeepGemmTimeoutProxy,
        )

        calls: list[int] = []
        proxy = _DeepGemmTimeoutProxy(SimpleNamespace(set_barrier_timeout_seconds=calls.append))

        proxy.set_timeout_seconds(1800)
        proxy.set_timeout_seconds(None)

        assert calls == [1800, 0]

    def test_deepgemm_proxy_sets_the_barrier_timeout(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("DeepGEMM's device runtime needs a GPU")
        dg = pytest.importorskip("tensorrt_llm.deep_gemm")
        from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm import (
            _DeepGemmTimeoutProxy,
        )

        proxy = _DeepGemmTimeoutProxy(dg)
        try:
            proxy.set_timeout_seconds(1800)
            assert dg.get_barrier_timeout_seconds() == 1800
            proxy.set_timeout_seconds(None)
            assert dg.get_barrier_timeout_seconds() == 0
            with pytest.raises(RuntimeError):
                dg.set_barrier_timeout_seconds(86401)
        finally:
            proxy.set_timeout_seconds(None)

    def test_deep_ep_proxy_sets_the_kernel_timeout(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("DeepEP is built for SM90 and newer")
        from tensorrt_llm._torch.moe.fused_moe import deep_ep_utils

        if not deep_ep_utils.deep_ep_installed:
            pytest.skip("DeepEP is not installed")
        proxy = deep_ep_utils._TIMEOUT_PROXY
        try:
            proxy.set_timeout_seconds(1800)
            assert deep_ep_utils.deep_ep.get_timeout_seconds() == 1800
            proxy.set_timeout_seconds(None)
            assert deep_ep_utils.deep_ep.get_timeout_seconds() == _DEEP_EP_DEFAULT_SEC
        finally:
            proxy.set_timeout_seconds(None)

    def test_nccl_ep_proxy_converts_seconds_to_group_nanoseconds(self) -> None:
        from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import _NcclEpGroupTimeoutProxy

        calls: list[int] = []
        proxy = _NcclEpGroupTimeoutProxy(SimpleNamespace(set_timeout_ns=calls.append))

        proxy.set_timeout_seconds(1800)
        proxy.set_timeout_seconds(None)

        assert calls == [1_800_000_000_000, 0]
