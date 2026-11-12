# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-layer dynamic EPLB replica planners, keyed by the ``planner`` config value.

``PER_LAYER_DYNAMIC_PLANNERS`` is the one place that maps a ``planner`` value
to its scheduler group and capabilities. The load balancer and its layers read
it; ``MoePerLayerDynamicLoadBalancerConfig.validate_expert_capacity`` and the
``TorchLlmArgs`` validator import it lazily. ``llm_args`` keeps the
``planner`` ``Literal``, whose values are this map's keys, and the
``auxiliary_sms`` bounds, which its field and the serve schema need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .halo_r_planner import build_halo_r_slot_scheduler_group, validate_halo_r_config
from .planner import RebalanceSlotSchedulerGroup, build_rebalance_slot_scheduler_group

__all__ = ["PER_LAYER_DYNAMIC_PLANNERS", "PerLayerDynamicPlanner"]


@dataclass(frozen=True)
class PerLayerDynamicPlanner:
    """What one per-layer dynamic EPLB replica planner supports.

    Attributes:
        build_group: Builds one layer's scheduler group from its manager's
            resources; every planner's builder takes the keywords of
            ``build_rebalance_slot_scheduler_group``.
        helper_multicast: Whether its copy stores through NVLink multicast
            objects bound to the helper sets.
        breakable_prefill_cuda_graphs: Whether breakable prefill CUDA graphs
            can capture its layer calls.
        validate_config: Rejects a ``num_slots``, expert count and EP size
            that the planner cannot plan, or ``None`` when every geometry the
            shared checks accept is supported. Called with keywords
            ``num_slots``, ``num_experts`` and ``ep_size``.
    """

    build_group: Callable[..., RebalanceSlotSchedulerGroup]
    helper_multicast: bool
    breakable_prefill_cuda_graphs: bool
    validate_config: Optional[Callable[..., None]] = None


PER_LAYER_DYNAMIC_PLANNERS: dict[str, PerLayerDynamicPlanner] = {
    "halo_q": PerLayerDynamicPlanner(
        build_group=build_rebalance_slot_scheduler_group,
        helper_multicast=True,
        breakable_prefill_cuda_graphs=False,
    ),
    "halo_r": PerLayerDynamicPlanner(
        build_group=build_halo_r_slot_scheduler_group,
        helper_multicast=False,
        breakable_prefill_cuda_graphs=True,
        validate_config=validate_halo_r_config,
    ),
}
