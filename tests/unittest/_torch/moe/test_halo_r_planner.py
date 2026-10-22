# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests of the HALO-R planner reference and its group selection."""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _torch.moe.halo_r_reference import plan_halo_r

from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
    PerLayerDynamicMoeLoadBalancer,
    halo_r_planner,
)
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.halo_r_planner import (
    HALO_R_SEND_COUNT,
    validate_halo_r_geometry,
)

pytestmark = pytest.mark.cpu_only

# Eight ranks of four experts. Rank 0 holds the hot expert 0 (300 of its 400
# routes); ranks 1, 4 and 5 are idle and the others sit at the balanced 100.
WORKED_EXAMPLE = [300, 40, 30, 30, 0, 0, 0, 0, 40, 30, 20, 10, 40, 30, 20, 10]
WORKED_EXAMPLE += [0, 0, 0, 0, 0, 0, 0, 0, 40, 30, 20, 10, 40, 30, 20, 10]


def test_worked_example_replicates_hot_expert_to_idle_ranks():
    plan = plan_halo_r(WORKED_EXAMPLE, ep=8, helper_slots=2, send_count=4)
    assert plan.target == 100
    assert len(plan.broadcasts) == 1
    broadcast = plan.broadcasts[0]
    assert (broadcast.expert, broadcast.owner) == (0, 0)
    assert broadcast.rows == {1: 100, 4: 100, 5: 100}
    assert broadcast.helper == {1: 0, 4: 0, 5: 0}
    assert plan.quota(WORKED_EXAMPLE, 8) == [[0, 100, 0, 0, 100, 100, 0, 0]]
    assert plan.load_after == [100] * 8


def test_receivers_fill_helper_slots_in_expert_order():
    histogram = [0, 0, 80, 10, 40, 200, 40, 10, 10, 80, 200, 0, 0, 80, 0, 200]
    plan = plan_halo_r(histogram, ep=4, helper_slots=4, send_count=4)
    assert [b.expert for b in plan.broadcasts] == [5, 10, 15]
    assert [b.helper[0] for b in plan.broadcasts] == [0, 1, 2]
    assert plan.helper_table[0] == [5, 10, 15, -1]
    words = plan.plan_words()
    stride = 384
    # Helper slot 2 sets the high mask bit of rank 0.
    assert words[2 + 3 * stride + 2] == 0
    assert words[2 + 5 * stride + 2] == 1


@pytest.mark.parametrize("seed", range(4))
def test_plans_respect_helper_budget_and_conserve_routes(seed):
    rng = random.Random(seed)
    for _ in range(50):
        ep = rng.choice([2, 4, 8, 16, 32])
        home = rng.randint(HALO_R_SEND_COUNT, 12)
        helper_slots = rng.randint(1, 4)
        histogram = [rng.randint(0, 300) if rng.random() < 0.7 else 0 for _ in range(ep * home)]
        histogram[rng.randrange(ep * home)] += rng.randint(0, 5000)
        plan = plan_halo_r(histogram, ep, helper_slots, HALO_R_SEND_COUNT)
        received = [0] * ep
        for b in plan.broadcasts:
            assert b.owner not in b.rows
            assert sum(b.rows.values()) <= histogram[b.expert]
            for rank, slot in b.helper.items():
                received[rank] += 1
                assert plan.helper_table[rank][slot] == b.expert
        assert max(received) <= helper_slots
        assert sum(plan.load_after) == sum(histogram)
        quotas = plan.quota(histogram, ep)
        for b, quota in zip(plan.broadcasts, quotas):
            assert sum(quota) == histogram[b.expert]


@pytest.mark.parametrize(
    "ep_size, home_experts, helper_slots, message",
    [
        pytest.param(1, 12, 2, "EP ranks", id="single-rank"),
        pytest.param(64, 6, 2, "EP ranks", id="above-one-warp"),
        pytest.param(8, 48, 5, "helper slots", id="five-helpers"),
        pytest.param(8, 3, 2, "experts per rank", id="fewer-home-experts-than-sends"),
        pytest.param(32, 16, 2, "at most 384 experts", id="too-many-experts"),
    ],
)
def test_geometry_validation_rejects_unsupported_shapes(
    ep_size, home_experts, helper_slots, message
):
    with pytest.raises(ValueError, match=message):
        validate_halo_r_geometry(
            ep_size=ep_size, home_experts=home_experts, helper_slots=helper_slots
        )


def test_geometry_validation_accepts_ep32_two_helper_slots():
    validate_halo_r_geometry(ep_size=32, home_experts=12, helper_slots=2)


class _Config:
    mode = "per_layer_dynamic"
    num_slots = 24
    auxiliary_sms = 8
    helper_slot_mode = "shared_banks"

    def __init__(self, planner):
        self.planner = planner

    def setup(self, ep_rank, ep_size):
        pass

    def validate_expert_capacity(self, num_experts):
        pass


class _Arena:
    home_experts = 4
    helper_slots = 2
    bundle = None

    def __init__(self):
        self.provider = SimpleNamespace(pool=None)


def _allocate_arena(layer, monkeypatch, arena):
    """Allocate ``arena`` through the layer handle, as the quant method does."""
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    monkeypatch.setattr(layer.manager, "get_ep_comm", lambda process_group: object())
    monkeypatch.setattr(layer.manager, "create_arena_provider", lambda: object())
    monkeypatch.setattr(arena_module, "allocate_rebalance_arena", MagicMock(return_value=arena))
    return layer.allocate_arena(
        bundle=arena.bundle,
        ep_process_group=object(),
        mapping=object(),
        device=0,
        layer_idx=layer.layer_idx,
    )


class _Backend:
    def __init__(self):
        self.tactic_autotune = False
        self.mapping = SimpleNamespace(local_rank=0)
        self.ep_process_group = object()
        self.max_tokens_per_rank_capacity = 256
        self.layer_idx = 3


def test_halo_r_planner_builds_the_halo_r_group(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner

    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=_Config("halo_r"))
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = SimpleNamespace(home_experts=4, helper_slots=2)
    halo_r_builder = MagicMock(return_value=group)
    halo_q_builder = MagicMock()
    monkeypatch.setattr(halo_r_planner, "build_halo_r_slot_scheduler_group", halo_r_builder)
    monkeypatch.setattr(planner, "build_rebalance_slot_scheduler_group", halo_q_builder)
    arena = _Arena()
    _allocate_arena(layer, monkeypatch, arena)
    backend = _Backend()

    layer.bind_backend(backend)

    halo_q_builder.assert_not_called()
    halo_r_builder.assert_called_once_with(
        arena=arena,
        manager=manager,
        mapping=backend.mapping,
        ep_process_group=backend.ep_process_group,
        home_experts=4,
        helper_slots=2,
        topk=2,
        max_tokens_per_rank=256,
        layer_idx=3,
    )
    assert layer._scheduler_group is group
    assert repr(manager).endswith("planner=halo_r)")


def test_default_planner_builds_the_halo_q_group(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner

    config = _Config("halo_q")
    del config.planner
    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    halo_r_builder = MagicMock()
    halo_q_builder = MagicMock(return_value=SimpleNamespace(home_experts=4, helper_slots=2))
    monkeypatch.setattr(halo_r_planner, "build_halo_r_slot_scheduler_group", halo_r_builder)
    monkeypatch.setattr(planner, "build_rebalance_slot_scheduler_group", halo_q_builder)

    _allocate_arena(layer, monkeypatch, _Arena())
    layer.bind_backend(_Backend())

    halo_r_builder.assert_not_called()
    halo_q_builder.assert_called_once()
    assert manager.planner == "halo_q"


def test_manager_rejects_unknown_planner():
    with pytest.raises(ValueError, match="unknown per-layer dynamic EPLB planner"):
        PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=_Config("halo_m"))


def test_manager_rejects_halo_r_with_resident_helper_slots():
    config = _Config("halo_r")
    config.helper_slot_mode = "per_layer_resident"
    with pytest.raises(ValueError, match="requires helper_slot_mode 'shared_banks'"):
        PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)


def test_reference_plan_words_have_the_kernel_layout():
    plan = plan_halo_r(WORKED_EXAMPLE, ep=8, helper_slots=2, send_count=4)
    words = plan.plan_words(epoch=7)
    assert len(words) == 2 + 6 * 384
    assert words[:2] == [7, 1]
    assert words[2 + 2 * 384] == (1 << 1) | (1 << 4) | (1 << 5)
