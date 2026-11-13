# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/gemm/router_half_pin on CPU: arming wraps the tp2ctx all-gather's full-shape router route once, sends only
qualified calls to the pinned GEMM and the rest to the stock route, and the kill switch leaves the route untouched."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.comm import tp2ctx_ipc
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    router_half_pin as pin,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def workspace(monkeypatch):
    """A stand-in for the IPC workspace class the arm patches, and the calls its stock route received."""
    calls = []

    class Workspace:
        def route_owned_full_shape(self, x, weight, rows):
            calls.append((self, x, weight, rows))
            return "stock"

    monkeypatch.setattr(tp2ctx_ipc, "Tp2ctxIpcWorkspace", Workspace)
    monkeypatch.setattr(pin, "_ARMED", False)
    monkeypatch.setattr(pin, "_DISABLED", False)
    return Workspace, calls


def test_kill_switch_prevents_arming_and_dispatch(workspace, monkeypatch):
    cls, calls = workspace
    monkeypatch.setattr(pin, "_DISABLED", True)
    stock = cls.route_owned_full_shape
    pin.arm_router_half_pin()
    assert cls.route_owned_full_shape is stock
    assert not pin._ARMED
    # The kill switch declines before touching tensor or CUDA properties.
    assert pin.try_router_half_pin(object(), object(), 16384) is None
    assert calls == []


def test_arming_is_idempotent_and_keeps_the_stock_route(workspace, monkeypatch):
    cls, calls = workspace
    pinned = object()
    dispatches = []

    def dispatch(x, weight, rows):
        dispatches.append((x, weight, rows))
        return pinned if rows == 16384 else None

    monkeypatch.setattr(pin, "try_router_half_pin", dispatch)
    pin.arm_router_half_pin()
    installed = cls.route_owned_full_shape
    pin.arm_router_half_pin()
    assert cls.route_owned_full_shape is installed
    instance = cls()
    x, weight = object(), object()
    assert instance.route_owned_full_shape(x, weight, 16384) is pinned
    assert calls == []
    assert instance.route_owned_full_shape(x, weight, 31) == "stock"
    assert calls == [(instance, x, weight, 31)]
    assert dispatches == [(x, weight, 16384), (x, weight, 31)]
