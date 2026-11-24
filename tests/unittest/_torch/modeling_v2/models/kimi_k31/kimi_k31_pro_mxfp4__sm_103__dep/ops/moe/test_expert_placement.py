# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 static expert placement from the engine's EP load balancer configuration (CPU): no configuration or no
assignments is the identity placement, a layer's ``initial_global_assignments`` names the expert of every physical
slot, and configurations the fused routed MoE does not serve (redundant slots, online updates, a layer's assignment
that is not a permutation or is missing) raise."""

import random

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe.expert_placement import (  # noqa: E501
    expert_slots,
    layer_slot_experts,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.routing import (
    static_placement_mismatch,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig, MoePerLayerDynamicLoadBalancerConfig

NUM_EXPERTS = 896


def _permutation(seed: int) -> list[int]:
    perm = list(range(NUM_EXPERTS))
    random.Random(seed).shuffle(perm)
    return perm


def test_no_configuration_is_the_identity() -> None:
    assert layer_slot_experts(None, 1, NUM_EXPERTS) is None


def test_no_assignments_is_the_identity() -> None:
    config = MoeLoadBalancerConfig(num_slots=NUM_EXPERTS)
    assert static_placement_mismatch(config, NUM_EXPERTS) is None
    assert layer_slot_experts(config, 1, NUM_EXPERTS) is None


def test_identity_assignment_is_the_identity() -> None:
    config = MoeLoadBalancerConfig(
        num_slots=NUM_EXPERTS, initial_global_assignments={1: list(range(NUM_EXPERTS))}
    )
    assert layer_slot_experts(config, 1, NUM_EXPERTS) is None


@pytest.mark.parametrize("world", [4, 16])
def test_assignment_maps_the_slots(world) -> None:
    assignments = {layer: _permutation(layer) for layer in (1, 2, 92)}
    config = MoeLoadBalancerConfig(num_slots=NUM_EXPERTS, initial_global_assignments=assignments)
    per_rank = NUM_EXPERTS // world
    for layer, perm in assignments.items():
        slot_experts = layer_slot_experts(config, layer, NUM_EXPERTS)
        assert slot_experts == tuple(perm)
        slots = expert_slots(slot_experts)
        assert sorted(slots) == list(range(NUM_EXPERTS))
        for slot, expert in enumerate(perm):
            assert slots[expert] == slot
        # Rank r loads the checkpoint experts of its slots, in slot order.
        for r in range(world):
            local = slot_experts[r * per_rank : (r + 1) * per_rank]
            assert [slots[e] // per_rank for e in local] == [r] * per_rank
            assert [slots[e] for e in local] == list(range(r * per_rank, (r + 1) * per_rank))


def test_assignment_from_a_yaml_file(tmp_path) -> None:
    """The YAML layout generate_eplb_config.py writes (integer layer keys, flow-style lists) loads through the
    engine's validator into the slot table."""
    from tensorrt_llm.llmapi.llm_args import MoeConfig, TorchLlmArgs

    perm = _permutation(7)
    path = tmp_path / "eplb.yaml"
    path.write_text(
        f"initial_global_assignments:\n  1: {perm}\nlayer_updates_per_iter: 0\nnum_slots: {NUM_EXPERTS}\n"
    )
    args = TorchLlmArgs(model="/tmp/dummy_model", moe_config=MoeConfig(load_balancer=str(path)))
    assert layer_slot_experts(args.moe_config.load_balancer, 1, NUM_EXPERTS) == tuple(perm)


@pytest.mark.parametrize(
    "kwargs, missed",
    [
        (dict(num_slots=NUM_EXPERTS + 16), "redundant"),
        (dict(num_slots=NUM_EXPERTS, layer_updates_per_iter=1), "layer_updates_per_iter=1"),
        (dict(), "num_slots=None"),
    ],
    ids=["redundant-slots", "online", "no-num-slots"],
)
def test_unserved_configuration_raises(kwargs, missed) -> None:
    config = MoeLoadBalancerConfig(**kwargs)
    assert missed in static_placement_mismatch(config, NUM_EXPERTS)
    with pytest.raises(ValueError, match=missed):
        layer_slot_experts(config, 1, NUM_EXPERTS)


def test_per_layer_dynamic_eplb_is_no_static_placement() -> None:
    """Per-layer dynamic EPLB places its helper copies per call; the static placement refuses its configuration (the
    target keeps the identity placement of the resident experts under it)."""
    config = MoePerLayerDynamicLoadBalancerConfig(num_slots=1024)
    assert "per-layer dynamic EPLB" in static_placement_mismatch(config, NUM_EXPERTS)
    with pytest.raises(ValueError, match="per-layer dynamic EPLB"):
        layer_slot_experts(config, 1, NUM_EXPERTS)


def test_redundant_assignment_raises() -> None:
    num_slots = NUM_EXPERTS + 16
    assignment = list(range(NUM_EXPERTS)) + list(range(16))
    config = MoeLoadBalancerConfig(num_slots=num_slots, initial_global_assignments={1: assignment})
    with pytest.raises(ValueError, match="redundant"):
        layer_slot_experts(config, 1, NUM_EXPERTS)


def test_non_permutation_raises() -> None:
    assignment = list(range(NUM_EXPERTS))
    assignment[5] = 4
    config = MoeLoadBalancerConfig(
        num_slots=NUM_EXPERTS, initial_global_assignments={1: assignment}
    )
    with pytest.raises(ValueError, match="not a permutation"):
        layer_slot_experts(config, 1, NUM_EXPERTS)


def test_missing_layer_raises() -> None:
    config = MoeLoadBalancerConfig(
        num_slots=NUM_EXPERTS, initial_global_assignments={1: _permutation(1)}
    )
    with pytest.raises(KeyError, match="layer_idx 2"):
        layer_slot_experts(config, 2, NUM_EXPERTS)
