# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 static expert placement from the engine's EP load balancer configuration (CPU): a layer's
``initial_global_assignments`` names the expert of every physical slot, and each rank loads the checkpoint experts
of its slots in slot order."""

import random

import pytest

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe.expert_placement import (  # noqa: E501
    expert_slots,
    layer_slot_experts,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS = 896


def _permutation(seed: int) -> list[int]:
    perm = list(range(NUM_EXPERTS))
    random.Random(seed).shuffle(perm)
    return perm


@pytest.mark.parametrize("world", [16])
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
