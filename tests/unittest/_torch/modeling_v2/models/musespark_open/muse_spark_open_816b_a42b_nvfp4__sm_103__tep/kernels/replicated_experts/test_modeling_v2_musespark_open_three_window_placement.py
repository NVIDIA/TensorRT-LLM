# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three ring windows (``three_window_placement``): its plan and its install wiring.

Host-only. The three-window placement modifies the replicated experts'
weight-load install (like the persistent experts): it builds on the replicated
experts (the balanced placement), the persistent experts (the persistent
kernel runs the 96-expert window) and the one-chain prologue (whose routed pool
kernel places over three windows). The install passes ``windows=3`` only when
it is planned; ``ring_buddy.install`` refuses a third window without the
persistent kernel or with the shadow mode; the layer state exposes the
64-expert prefix (the two-window stack) beside the whole stack; and the one-chain
prologue asks its kernel for ``PLACE_BALANCE3`` on a three-window layer only.
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    weights as target_weights,
)

pytestmark = pytest.mark.cpu_only


def test_three_windows_build_on_the_persistent_experts_and_the_prologue():
    assert dispatch.closure({"three_window_placement"}) >= {
        "replicated_experts",
        "persistent_experts",
        "moe_prologue",
    }
    with pytest.raises(ValueError, match="builds on"):
        dispatch.Plan(
            frozenset({"three_window_placement", "replicated_experts", "persistent_experts"})
        )


@pytest.mark.parametrize(
    "paths, extra",
    [
        (("replicated_experts", "persistent_experts", "moe_prologue"), {"persistent": True}),
        (("three_window_placement",), {"persistent": True, "windows": 3}),
        (("replicated_experts",), {}),
    ],
)
def test_the_install_passes_a_third_window_only_with_three_windows(monkeypatch, paths, extra):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )

    calls = []
    monkeypatch.setattr(ring_buddy, "install", lambda *a, **k: calls.append(k))
    target_weights.install_loaded_features(object(), object(), dispatch.plan_of(*paths))
    assert len(calls) == 1
    got = {k: v for k, v in calls[0].items() if k in ("persistent", "windows")}
    assert got == extra


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(windows=3), "needs persistent=True"),
        (dict(windows=3, persistent=True, balance=False), "balanced placement"),
        (
            dict(windows=4, persistent=True),
            "two ring windows, or three with three_window_placement",
        ),
    ],
)
def test_ring_buddy_install_refuses_a_third_window_it_cannot_place(kwargs, message):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )

    with pytest.raises(ValueError, match=message):
        ring_buddy.install(types.SimpleNamespace(layers=[]), object(), **kwargs)


def test_the_layer_state_keeps_the_two_window_stack_as_the_64_expert_prefix():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )

    stacks = {"w3_w1_weight": torch.arange(96 * 4).reshape(96, 4)}
    state = ring_buddy._State("nvfp4", stacks, rank=3, balance=True, windows=3)
    assert sorted(state.by_window) == [64, 96]
    assert state.by_window[96] is stacks
    prefix = state.by_window[64]["w3_w1_weight"]
    assert prefix.shape == (64, 4) and prefix.data_ptr() == stacks["w3_w1_weight"].data_ptr()
    two = ring_buddy._State("nvfp4", {"w3_w1_weight": torch.zeros(64, 4)}, rank=0, balance=True)
    assert two.windows == 2 and list(two.by_window) == [64]


def test_the_prologue_asks_for_three_windows_on_a_three_window_layer():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.moe_prologue import (  # noqa: E501
        prologue as moe_prologue,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
        pool_gather_route as route,
    )

    def moe(**state):
        values = dict(active=True, balance=True, static_from=32, windows=2)
        values.update(state)
        return types.SimpleNamespace(_replicated_experts=types.SimpleNamespace(**values))

    assert moe_prologue._placement(moe(windows=3))[0] == route.PLACE_BALANCE3
    assert moe_prologue._placement(moe())[0] == route.PLACE_BALANCE
    assert moe_prologue._placement(moe(windows=3, balance=False, static_from=32))[0] == (
        route.PLACE_OWNER
    )
    assert moe_prologue._placement(moe(windows=3, active=False))[0] == route.PLACE_NONE


def test_the_routed_reference_places_three_windows_like_ring_windows():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        windows as ring_windows,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
        pool_gather_route as route,
    )

    assert route.PLACE_BALANCE3 not in (
        route.PLACE_NONE,
        route.PLACE_OWNER,
        route.PLACE_BALANCE,
        route.PLACE_STATIC,
    )
    assert any(cell.spec["place"] == route.PLACE_BALANCE3 for cell in route.PoolGatherRoute.CELLS)
    gen = torch.Generator().manual_seed(5)
    selected = torch.stack([torch.randperm(256, generator=gen)[:8] for _ in range(8)]).int()
    for rank in range(8):
        flat, stats = ring_windows.place(selected.reshape(-1).tolist(), rank, 3)
        assert max(stats[8:]) == ring_windows.smallest_max(stats[:8], 3)
        assert all(0 <= v < 256 for v in flat)
