# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests of the HALO-R planner reference and its group selection."""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
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


def test_worked_example_replicates_hot_expert_to_idle_ranks() -> None:
    plan = plan_halo_r(WORKED_EXAMPLE, ep=8, helper_slots=2, send_count=4)
    assert plan.target == 100
    assert len(plan.broadcasts) == 1
    broadcast = plan.broadcasts[0]
    assert (broadcast.expert, broadcast.owner) == (0, 0)
    assert broadcast.rows == {1: 100, 4: 100, 5: 100}
    assert broadcast.helper == {1: 0, 4: 0, 5: 0}
    assert plan.quota(WORKED_EXAMPLE, 8) == [[0, 100, 0, 0, 100, 100, 0, 0]]
    assert plan.load_after == [100] * 8


def test_receivers_fill_helper_slots_in_expert_order() -> None:
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


def _moves(plan) -> list:
    return [(b.expert, b.owner, b.rows, b.helper) for b in plan.broadcasts]


def test_receiver_is_the_lowest_ranked_one_with_room_not_the_roomiest() -> None:
    """Written out by hand: 4 ranks of 4 experts, 81 routes, so the target is ceil(81 / 4) = 21.

    Loads are [24, 26, 16, 15]: ranks 2 and 3 have room 5 and 6. Rank 1's
    hottest expert 4 (13 routes, the lower of its two tied experts) wins with
    min(6, 13, excess 5) = 5 rows. Both receivers have room for 5; the lower
    rank, 2, takes them, not rank 3 with more room. Then rank 0's expert 0
    moves min(6, 13, excess 3) = 3 rows to rank 3. The plan lists expert 0
    before expert 4 although it moved second.
    """
    histogram = [13, 3, 0, 8, 13, 0, 0, 13, 5, 3, 8, 0, 8, 0, 2, 5]
    plan = plan_halo_r(histogram, ep=4, helper_slots=2, send_count=4)
    assert plan.target == 21
    assert _moves(plan) == [(0, 0, {3: 3}, {3: 0}), (4, 1, {2: 5}, {2: 0})]
    assert plan.load_after == [21, 21, 21, 18]


def test_rows_stop_at_the_owners_excess_over_a_rounded_up_target() -> None:
    """Written out by hand: 4 ranks of 4 experts, 58 routes, so the target is ceil(58 / 4) = 15.

    Loads are [2, 29, 9, 18]. Rank 1 (excess 14) moves its expert 4 (8 rows,
    the lowest of three tied experts) to rank 0, the only rank with room for
    8; then its expert 5 sends min(room 6, 8, excess 6) = 6 rows to rank 2.
    Rank 3's expert 15 then moves only its excess, 3 rows, although rank 0
    has room for 5 and the expert has 8 routes.
    """
    histogram = [0, 0, 2, 0, 8, 8, 8, 5, 1, 0, 8, 0, 5, 5, 0, 8]
    plan = plan_halo_r(histogram, ep=4, helper_slots=2, send_count=4)
    assert plan.target == 15
    assert _moves(plan) == [
        (4, 1, {0: 8}, {0: 0}),
        (5, 1, {2: 6}, {2: 0}),
        (15, 3, {0: 3}, {0: 1}),
    ]
    assert plan.load_after == [13, 15, 15, 15]


@pytest.mark.parametrize("seed", range(4))
def test_plans_respect_helper_budget_and_conserve_routes(seed) -> None:
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
    "ep_size, home_experts, helper_slots, auxiliary_sms, message",
    [
        pytest.param(1, 12, 2, 4, "EP ranks", id="single-rank"),
        pytest.param(6, 12, 2, 4, None, id="even-ep"),
        pytest.param(7, 12, 2, 4, "even number", id="odd-ep"),
        pytest.param(64, 6, 2, 4, "EP ranks", id="above-one-warp"),
        pytest.param(8, 48, 5, 4, "helper slots", id="five-helpers"),
        pytest.param(8, 3, 2, 4, "experts per rank", id="fewer-home-experts-than-sends"),
        pytest.param(32, 16, 2, 4, "at most 384 experts", id="too-many-experts"),
        pytest.param(8, 48, 2, 1, "auxiliary_sms", id="one-cta"),
    ],
)
def test_geometry_validation_rejects_unsupported_shapes(
    ep_size, home_experts, helper_slots, auxiliary_sms, message
) -> None:
    def validate():
        validate_halo_r_geometry(
            ep_size=ep_size,
            home_experts=home_experts,
            helper_slots=helper_slots,
            auxiliary_sms=auxiliary_sms,
        )

    if message is None:
        validate()
    else:
        with pytest.raises(ValueError, match=message):
            validate()


def test_geometry_validation_accepts_ep32_two_helper_slots() -> None:
    validate_halo_r_geometry(ep_size=32, home_experts=12, helper_slots=2, auxiliary_sms=2)


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
        self.flags = torch.empty(4, dtype=torch.uint64)

    def terminal_flags_tensor(self):
        return self.flags


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


def test_halo_r_planner_builds_the_halo_r_group(monkeypatch) -> None:
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


def test_default_planner_builds_the_halo_q_group(monkeypatch) -> None:
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


def test_rebinding_the_backend_forgets_helper_residency(monkeypatch) -> None:
    """Loaders bind the layer again after they rewrite its weights."""
    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=_Config("halo_r"))
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = SimpleNamespace(home_experts=4, helper_slots=2, forget_helper_residency=MagicMock())
    monkeypatch.setattr(
        halo_r_planner, "build_halo_r_slot_scheduler_group", MagicMock(return_value=group)
    )
    _allocate_arena(layer, monkeypatch, _Arena())
    backend = _Backend()

    layer.bind_backend(backend)
    group.forget_helper_residency.assert_not_called()
    layer.bind_backend(backend)
    layer.bind_backend(backend)
    assert group.forget_helper_residency.call_count == 2

    # A producer without the hook, such as HALO-Q's, rebinds as before.
    del group.forget_helper_residency
    layer.bind_backend(backend)


def test_manager_rejects_unknown_planner() -> None:
    with pytest.raises(ValueError, match="unknown per-layer dynamic EPLB planner"):
        PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=_Config("halo_m"))


def test_manager_keeps_halo_r_helper_slots_resident_per_layer() -> None:
    config = _Config("halo_r")
    config.helper_slot_mode = "per_layer_resident"
    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)
    assert manager.helper_residency


@pytest.mark.parametrize("helper_slot_mode", ["per_layer_resident", "shared_banks"])
@pytest.mark.parametrize(("planner_name", "multicast"), [("halo_q", True), ("halo_r", False)])
def test_only_halo_q_helper_sets_get_multicast(planner_name, multicast, helper_slot_mode) -> None:
    config = _Config(planner_name)
    config.helper_slot_mode = helper_slot_mode
    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)
    provider = manager.create_arena_provider()
    assert provider._helper_multicast is multicast
    assert provider._per_layer_helper_sets is (helper_slot_mode == "per_layer_resident")


def test_reference_keeps_resident_experts_in_their_slots() -> None:
    histogram = [0, 0, 80, 10, 40, 200, 40, 10, 10, 80, 200, 0, 0, 80, 0, 200]
    ascending = plan_halo_r(histogram, ep=4, helper_slots=4, send_count=4)
    assert ascending.helper_table[0] == [5, 10, 15, -1]
    unknown = [[-1] * 4 for _ in range(4)]
    assert plan_halo_r(histogram, 4, 4, 4, residency=unknown) == ascending
    # Experts 15 and 10 stay where they are (10 in the lowest slot holding it);
    # expert 5 takes the lowest free slot.
    residency = [[-1, 15, 10, 10]] + unknown[1:]
    plan = plan_halo_r(histogram, 4, 4, 4, residency=residency)
    assert plan.helper_table[0] == [5, 15, 10, -1]
    assert [b.helper[0] for b in plan.broadcasts] == [0, 2, 1]
    assert [(b.expert, b.rows) for b in plan.broadcasts] == [
        (b.expert, b.rows) for b in ascending.broadcasts
    ]
    assert plan.load_after == ascending.load_after


def test_reference_plan_words_have_the_kernel_layout() -> None:
    plan = plan_halo_r(WORKED_EXAMPLE, ep=8, helper_slots=2, send_count=4)
    words = plan.plan_words(epoch=7)
    assert len(words) == 2 + 6 * 384
    assert words[:2] == [7, 1]
    assert words[2 + 2 * 384] == (1 << 1) | (1 << 4) | (1 << 5)


def test_one_capability_map_covers_every_planner() -> None:
    """The load balancer, its layers and the config validator read one planner map."""
    import importlib
    import typing

    from tensorrt_llm.llmapi.llm_args import (
        _PER_LAYER_DYNAMIC_PLANNERS,
        MoePerLayerDynamicLoadBalancerConfig,
    )

    field = MoePerLayerDynamicLoadBalancerConfig.model_fields["planner"]
    assert set(typing.get_args(field.annotation)) == set(_PER_LAYER_DYNAMIC_PLANNERS)
    for spec in _PER_LAYER_DYNAMIC_PLANNERS.values():
        module_name, _, function_name = spec.group_builder.partition(":")
        assert callable(getattr(importlib.import_module(module_name), function_name))
    # HALO-Q's copy stores through multicast; only HALO-R can be captured.
    multicast = {name: spec.helper_multicast for name, spec in _PER_LAYER_DYNAMIC_PLANNERS.items()}
    assert multicast == {"halo_q": True, "halo_r": False}
    capturable = [
        name
        for name, spec in _PER_LAYER_DYNAMIC_PLANNERS.items()
        if spec.breakable_prefill_cuda_graphs
    ]
    assert capturable == ["halo_r"]


def test_real_halo_r_config_builds_the_halo_r_group_through_the_factory(monkeypatch) -> None:
    """The real config, through the model factory, selects HALO-R end to end on the host."""
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import maybe_create_moe_load_balancer
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner
    from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig

    config = MoePerLayerDynamicLoadBalancerConfig(
        num_slots=24, planner="halo_r", helper_slot_mode="shared_banks"
    )
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=4, moe_cluster_size=1)
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        moe_load_balancer=config,
    )
    with maybe_create_moe_load_balancer(model_config, mapping) as manager:
        assert isinstance(manager, PerLayerDynamicMoeLoadBalancer)
        # The unset auxiliary_sms resolves to HALO-R's default; its helper sets get no multicast.
        assert (manager.planner, manager.auxiliary_sms) == ("halo_r", 16)
        assert not manager.planner_spec.helper_multicast
        layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
        group = SimpleNamespace(home_experts=4, helper_slots=2)
        halo_r_builder = MagicMock(return_value=group)
        halo_q_builder = MagicMock()
        monkeypatch.setattr(halo_r_planner, "build_halo_r_slot_scheduler_group", halo_r_builder)
        monkeypatch.setattr(planner, "build_rebalance_slot_scheduler_group", halo_q_builder)
        _allocate_arena(layer, monkeypatch, _Arena())
        layer.bind_backend(_Backend())
        halo_q_builder.assert_not_called()
        halo_r_builder.assert_called_once()
        assert layer._scheduler_group is group


def test_config_limits_match_the_planner_module() -> None:
    """llm_args validates HALO-R configs against the same limits the planner module enforces."""
    from tensorrt_llm.llmapi import llm_args

    assert llm_args._HALO_R_MAX_HELPER_SLOTS == halo_r_planner.HALO_R_MAX_HELPER_SLOTS
    assert llm_args._HALO_R_MIN_HOME_EXPERTS == HALO_R_SEND_COUNT
    assert llm_args._HALO_R_MAX_EP == halo_r_planner.HALO_R_MAX_EP
    assert llm_args._HALO_R_MIN_AUXILIARY_SMS == halo_r_planner.HALO_R_MIN_AUXILIARY_SMS
