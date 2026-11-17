# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel load balancing (EPLB): slot layouts, layer registration and routing.

The engine owns the load balancer: the model loader creates it around the model
(`maybe_create_moe_load_balancer`) and finalizes it after the weights load, and
the executor drives it on every forward. A target takes part in three steps,
and this module is what they share:

1. `eplb_slots`: the slots of one MoE layer on this rank, that is which expert
   each local slot holds. The target allocates and loads its expert weights by
   slot. Without EPLB every expert has one slot and a rank holds a contiguous
   window of them, so a target can read its experts through `EplbSlots` either
   way.
2. `EplbLayer.register`: register the layer with the engine's balancer, at
   construction and in the order the layers run.
3. `EplbLayer.route`: map the router's expert ids to slot ids before the expert
   GEMMs, which then take `num_slots` as their expert count and this rank's
   slot window as their local experts.

`eplb_mode` says whether the engine runs EPLB for a deployment and in which
mode. A target lists the modes it implements in `EPLB_MODES`, and the resolver
declines EPLB deployments in any other mode (see the Modeling V2 README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2._router_index import moe_eplb_mode
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    SingleLayerMoeLoadBalancer,
    get_moe_load_balancer,
)


def eplb_mode(model_config) -> Optional[str]:
    """The EPLB mode the engine runs this deployment with: ``static``, ``online``,
    ``per_layer_dynamic``, or None when it creates no load balancer."""
    if getattr(model_config, "moe_load_balancer", None) is None:
        return None
    return moe_eplb_mode(model_config)


@dataclass(frozen=True)
class EplbSlots:
    """One MoE layer's expert slots on this rank.

    Attributes:
        num_experts: The layer's experts.
        num_slots: Slots over all EP ranks; ``num_experts`` without EPLB.
        slot_offset: The global id of this rank's first slot.
        slot_experts: The expert each of this rank's slots holds, in slot order.
        global_assignments: The expert each global slot holds, in slot order.
    """

    num_experts: int
    num_slots: int
    slot_offset: int
    slot_experts: tuple[int, ...]
    global_assignments: tuple[int, ...]

    @property
    def local_slots(self) -> int:
        """Slots on this rank, ``num_slots / ep_size``."""
        return len(self.slot_experts)

    @classmethod
    def from_assignments(
        cls, assignments: Sequence[int], num_experts: int, ep_size: int, ep_rank: int
    ) -> "EplbSlots":
        """The slots of ``ep_rank`` when global slot ``j`` holds expert ``assignments[j]``.

        Raises:
            ValueError: If the slots do not split evenly over the EP ranks, or do not
                hold every expert at least once and nothing else.
        """
        num_slots = len(assignments)
        if num_slots % ep_size != 0:
            raise ValueError(f"{num_slots} slots do not split evenly over {ep_size} EP ranks")
        if sorted(set(assignments)) != list(range(num_experts)):
            raise ValueError(
                f"the {num_slots} slots must hold every one of the {num_experts} experts "
                "at least once, and nothing else"
            )
        local = num_slots // ep_size
        offset = ep_rank * local
        return cls(
            num_experts=num_experts,
            num_slots=num_slots,
            slot_offset=offset,
            slot_experts=tuple(assignments[offset : offset + local]),
            global_assignments=tuple(assignments),
        )

    @classmethod
    def without_eplb(cls, num_experts: int, ep_size: int, ep_rank: int) -> "EplbSlots":
        """One slot per expert, each rank holding an equal contiguous window."""
        return cls.from_assignments(list(range(num_experts)), num_experts, ep_size, ep_rank)

    @classmethod
    def from_config(
        cls, config, layer_idx: int, num_experts: int, ep_size: int, ep_rank: int
    ) -> "EplbSlots":
        """The placement standard EPLB starts ``layer_idx`` with.

        ``config.initial_global_assignments[layer_idx]`` when the config gives
        assignments, which then has to name the layer as the built-in MoE
        requires; otherwise the built-in MoE's default, where rank ``r``'s slots
        hold the experts from ``r * num_experts // ep_size`` on, wrapping.
        """
        assignments = config.get_layer_initial_global_assignments(layer_idx)
        if assignments is None:
            if config.num_slots % ep_size != 0:
                raise ValueError(
                    f"{config.num_slots} slots do not split evenly over {ep_size} EP ranks"
                )
            local = config.num_slots // ep_size
            assignments = [
                (rank * num_experts // ep_size + slot) % num_experts
                for rank in range(ep_size)
                for slot in range(local)
            ]
        return cls.from_assignments(assignments, num_experts, ep_size, ep_rank)


def eplb_slots(model_config, layer_idx: int, num_experts: int) -> EplbSlots:
    """The slots of MoE layer ``layer_idx`` on this rank, with or without EPLB."""
    mapping = model_config.mapping
    if eplb_mode(model_config) is None:
        return EplbSlots.without_eplb(num_experts, mapping.moe_ep_size, mapping.moe_ep_rank)
    return EplbSlots.from_config(
        model_config.moe_load_balancer,
        layer_idx,
        num_experts,
        mapping.moe_ep_size,
        mapping.moe_ep_rank,
    )


class EplbRoute(OpWrapper):
    """Map each token's expert ids to the global slots that hold those experts.

    `token_selected_experts` is `[T, top_k]` int32 global expert ids, and
    `balancer_layer` the layer's handle in the engine's load balancer. The
    result is `[T, top_k]` int32 global slot ids under the layer's current
    placement.

    An expert held by one slot always maps to it. An expert with replicas maps
    to one of them: the kernel spreads an expert's occurrences over its
    replicas, and which replica it picks depends only on the input and, with
    `offset_by_ep_rank`, on the rank. Ranks that route the same gathered tokens
    with `offset_by_ep_rank=False` therefore agree on every slot, which a target
    that gathers its tokens before routing needs. An id outside
    `[0, num_experts)` maps to `num_slots`, a slot no rank holds.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="no replica: every expert held once, the kernel's direct lookup",
            spec=dict(experts=256, slots=256, ep_size=4, top_k=8, tokens=64),
        ),
        Cell(
            why="replicas: 288 slots over 256 experts at EP 4, 72 slots a rank",
            spec=dict(experts=256, slots=288, ep_size=4, top_k=8, tokens=256),
        ),
        Cell(
            why="more ids than one kernel block arbitrates (1024), so replicas are "
            "spread per block",
            spec=dict(experts=256, slots=288, ep_size=8, top_k=8, tokens=2048),
        ),
        Cell(
            why="an expert held twice on the same rank, and a hot expert held four times",
            spec=dict(experts=16, slots=32, ep_size=4, top_k=4, tokens=96, hot_replicas=4),
        ),
    )

    def __call__(
        self,
        token_selected_experts: torch.Tensor,
        balancer_layer: SingleLayerMoeLoadBalancer,
        offset_by_ep_rank: bool = False,
    ) -> torch.Tensor:
        """Route `token_selected_experts` `[T, top_k]` to global slot ids `[T, top_k]`."""
        return torch.ops.trtllm.moe_load_balance_routing(
            token_selected_experts,
            offset_by_ep_rank,
            balancer_layer.single_layer_load_balancer_ptr,
        )

    def reference(
        self,
        token_selected_experts: torch.Tensor,
        balancer_layer: SingleLayerMoeLoadBalancer,
        offset_by_ep_rank: bool = False,
    ) -> torch.Tensor:
        """The first slot that holds each expert, under the layer's current placement.

        Which replica the kernel picks is its own business (see the class
        docstring), so the test holds the op to the reference's *expert*, and
        holds the slot itself exactly only where an expert has one slot.
        `offset_by_ep_rank` changes only the replica, so it is ignored here.
        """
        placement = balancer_layer.single_layer_load_balancer_impl.get_old_rank_expert_ids()
        slot_expert = torch.tensor(
            [expert for rank in placement for expert in rank], dtype=torch.int64
        )
        num_slots = slot_expert.numel()
        num_experts = balancer_layer.expert_count
        first_slot = torch.full((num_experts,), num_slots, dtype=torch.int64)
        for slot in range(num_slots - 1, -1, -1):
            first_slot[slot_expert[slot]] = slot
        ids = token_selected_experts.cpu().to(torch.int64)
        out = torch.full_like(ids, num_slots)
        valid = (ids >= 0) & (ids < num_experts)
        out[valid] = first_slot[ids[valid]]
        return out.to(torch.int32).to(token_selected_experts.device)


eplb_route = EplbRoute()


class EplbLayer:
    """A MoE layer registered with the engine's load balancer, and its routing.

    Built with `register` while the target is constructed, inside the model
    loader's load-balancer context, once per MoE layer and in the order the
    layers run.

    Attributes:
        balancer_layer: The layer's handle in the engine's load balancer.
        slots: The layer's slots on this rank.
    """

    def __init__(self, balancer_layer: SingleLayerMoeLoadBalancer, slots: EplbSlots):
        self.balancer_layer = balancer_layer
        self.slots = slots

    @classmethod
    def register(cls, slots: EplbSlots, top_k: int, repeat_count: int = 1) -> "EplbLayer":
        """Add the layer to the engine's load balancer with its initial placement.

        Args:
            slots: The layer's slots on this rank; its `global_assignments` are the
                initial placement.
            top_k: Experts each token selects.
            repeat_count: How many times one forward runs the layer, such as the
                draft steps of a one-model MTP layer.

        Raises:
            RuntimeError: If no load balancer is active, which means the layer is
                built outside the model loader while the deployment runs EPLB.
        """
        balancer = get_moe_load_balancer()
        if balancer is None:
            raise RuntimeError(
                "this deployment runs EPLB, but no MoE load balancer is active: register "
                "the MoE layers while the model loader builds the model"
            )
        if repeat_count > 1:
            balancer.set_repeated_for_next_layer(repeat_count)
        balancer_layer = balancer.add_layer(slots.num_experts, top_k, slots.local_slots)
        balancer_layer.set_initial_weight_assignments(list(slots.global_assignments))
        return cls(balancer_layer, slots)

    def route(self, token_selected_experts: torch.Tensor) -> torch.Tensor:
        """Global expert ids `[T, top_k]` int32 -> global slot ids `[T, top_k]` int32.

        Routes with `offset_by_ep_rank=False`, so every rank that routes the same
        tokens gets the same slots.
        """
        return eplb_route(token_selected_experts, self.balancer_layer, False)
