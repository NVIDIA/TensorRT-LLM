# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 static expert placement for the expert-parallel routed MoE (pure EP under attention DP).

With the default placement rank ``r`` holds experts ``[r * E / W, (r + 1) * E / W)``; routed traffic is skewed, so the
rank holding the busiest experts sets every MoE layer's time. A placement puts each routed (logical) expert on a
physical slot ``rank * (E / W) + slot``, balancing the routed rows per rank.

The placement is the engine's static EP load balancer configuration (``moe_config.load_balancer``, a
``MoeLoadBalancerConfig``): per MoE layer, ``initial_global_assignments[layer][p]`` is the expert held by physical
slot ``p``. ``examples/wide_ep/ep_load_balancer/generate_eplb_config.py`` writes it from routing statistics. The
fused routed MoE (:mod:`.w4a8_ep_moe_fused`) holds one slot per expert (``num_slots`` = the expert count) and no
online updates (``layer_updates_per_iter`` 0). It dispatches each routed expert to its slot through the inverse table
and keeps the combine's grouping by the identity placement's owner ranks, so its output does not change by a bit; the
weights of physical slot ``p`` are those of checkpoint expert ``initial_global_assignments[layer][p]``. Routing checks
the configuration (``static_placement_mismatch``, in the family's ``routing.py``) before this target is built.
"""

from __future__ import annotations

from typing import Optional, Sequence

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.routing import static_placement_mismatch


def layer_slot_experts(config, layer_idx: int, num_experts: int) -> Optional[tuple[int, ...]]:
    """Physical slot -> expert table of MoE layer ``layer_idx`` from the load balancer configuration ``config``
    (``initial_global_assignments``, a permutation of the experts for every MoE layer, as routing admits it), or
    ``None`` for the identity placement: no configuration, no assignments, or assignments that are the identity.
    Raises ``ValueError`` on a configuration the fused routed MoE does not serve and ``KeyError`` when the
    assignments leave the layer out."""
    if config is None:
        return None
    if (why := static_placement_mismatch(config, num_experts)) is not None:
        raise ValueError(why)
    slot_experts = config.get_layer_initial_global_assignments(layer_idx)
    if slot_experts is None:
        return None
    slot_experts = tuple(int(e) for e in slot_experts)
    if slot_experts == tuple(range(num_experts)):
        return None
    return slot_experts


def expert_slots(slot_experts: Sequence[int]) -> tuple[int, ...]:
    """Each expert's physical slot under the slot -> expert table ``slot_experts`` (its inverse)."""
    slots = [0] * len(slot_experts)
    for slot, expert in enumerate(slot_experts):
        slots[expert] = slot
    return tuple(slots)


__all__ = ["expert_slots", "layer_slot_experts"]
