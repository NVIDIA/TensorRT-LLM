# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 router pin: full-shape stock bytes on both ownership slices, eager and graph."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm import m31_router_half_pin as pin

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Requires SM107",
)


@pytest.fixture(autouse=True)
def dispatch_state(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(pin, "_DISABLED", False)
    monkeypatch.setattr(pin, "_PREPARED", {})
    monkeypatch.setattr(pin, "_LOGGED", set())
    assert pin._load(), "SM107 CI must ship the native router pin operator"


@pytest.mark.parametrize("total_rows", sorted(pin.QUALIFIED_ROWS))
@pytest.mark.parametrize("owner", [0, 1])
def test_router_half_pin_exact_eager_and_graph(total_rows, owner):
    split = min(max((total_rows // 2 + 64) // 128 * 128, 128), total_rows - 1)
    start, rows = (0, split) if owner == 0 else (split, total_rows - split)
    generator = torch.Generator(device="cuda").manual_seed(20623 + total_rows + owner)
    x = torch.randn(rows, 6144, device="cuda", generator=generator)
    weight = torch.randn(128, 6144, device="cuda", generator=generator)
    full = torch.zeros(total_rows, 6144, device="cuda")

    def reference():
        full[start : start + rows].copy_(x)
        return torch.nn.functional.linear(full, weight)[start : start + rows]

    expected = reference()
    actual = pin.try_router_half_pin(x, weight, total_rows)
    assert actual is not None, "Qualified native descriptor unexpectedly declined"
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = pin.try_router_half_pin(x, weight, total_rows)
    assert actual is not None
    for _ in range(3):
        # Change replay inputs as well: stale eager results cannot satisfy this oracle.
        x.mul_(0.5)
        weight.add_(0.015625)
        expected = reference()
        graph.replay()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_router_half_pin_guards(monkeypatch):
    x = torch.empty(8192, 6144, device="cuda")
    weight = torch.empty(128, 6144, device="cuda")
    monkeypatch.setattr(pin, "_DISABLED", True)
    assert pin.try_router_half_pin(x, weight, 16384) is None
    monkeypatch.setattr(pin, "_DISABLED", False)
    assert pin.try_router_half_pin(x, weight, 16383) is None
    assert pin.try_router_half_pin(x[:-1], weight, 16384) is None
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    assert pin.try_router_half_pin(x, weight, 16384) is None
    assert pin._PREPARED == {}, "Declined inputs must not prepare a native descriptor"
