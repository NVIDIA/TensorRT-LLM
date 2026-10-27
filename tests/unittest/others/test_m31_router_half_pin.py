# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests of the production router arm, with only GPU/IPC dependencies mocked."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[3]
_CATALOG = "tensorrt_llm._torch._experimental.modeling_v2.catalog"
_SOURCE = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/gemm/m31_router_half_pin.py"
)


@pytest.fixture
def router(monkeypatch):
    calls = []

    class Workspace:
        def route_owned_full_shape(self, x, weight, rows):
            calls.append((self, x, weight, rows))
            return "stock"

    ipc = ModuleType(_CATALOG + ".comm.tp2ctx_ipc")
    ipc.Tp2ctxIpcWorkspace = Workspace
    monkeypatch.setitem(sys.modules, ipc.__name__, ipc)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.delenv("TRTLLM_M31_DISABLE_ROUTER_HALF_PIN", raising=False)

    def load():
        spec = importlib.util.spec_from_file_location(
            _CATALOG + ".gemm._router_pin_under_test", _SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load, Workspace, calls


def test_kill_switch_prevents_arming_and_dispatch(router, monkeypatch):
    load, workspace, calls = router
    monkeypatch.setenv("TRTLLM_M31_DISABLE_ROUTER_HALF_PIN", "1")
    pin = load()
    stock = workspace.route_owned_full_shape
    pin.arm_router_half_pin()
    assert workspace.route_owned_full_shape is stock
    assert not pin._ARMED
    # The kill switch must decline before touching tensor or CUDA properties.
    assert pin.try_router_half_pin(object(), object(), 16384) is None
    assert calls == []


def test_default_arm_is_idempotent_and_preserves_fallback(router, monkeypatch):
    load, workspace, calls = router
    pin = load()
    assert not pin._DISABLED
    pinned = object()
    dispatches = []

    def dispatch(x, weight, rows):
        dispatches.append((x, weight, rows))
        return pinned if rows == 16384 else None

    monkeypatch.setattr(pin, "try_router_half_pin", dispatch)
    pin.arm_router_half_pin()
    installed = workspace.route_owned_full_shape
    pin.arm_router_half_pin()
    assert workspace.route_owned_full_shape is installed
    instance = workspace()
    x, weight = object(), object()
    assert instance.route_owned_full_shape(x, weight, 16384) is pinned
    assert calls == []
    assert instance.route_owned_full_shape(x, weight, 31) == "stock"
    assert calls == [(instance, x, weight, 31)]
    assert dispatches == [(x, weight, 16384), (x, weight, 31)]
