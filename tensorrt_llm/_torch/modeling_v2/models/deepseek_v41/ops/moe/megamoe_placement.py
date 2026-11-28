# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A target's MegaMoE layers under the engine's warmup expert placement (`moe_config.load_balancer.warmup_placement`).

The engine's `WarmupPlacementController` (`moe/fused_moe/moe_warmup_placement.py`) learns the placement from the
served traffic and re-places experts at executor-step boundaries: it is created with the engine's MoE load balancer
when the config carries `warmup_placement`, and the executor drives it (`on_executor_step`, `set_live_rows`). The
engine's own MoE modules register with it from their constructors; this module registers a target's layer the same way
(not an entry: engine wiring, like `megamoe_workspace.megamoe_ep_process_group`):

* `megamoe_placement_register` (constructor time, every rank in the same layer order): the layer joins the controller
  as a static balancer layer (one copy per expert, or a replicated cover with extra slots) and gets a
  `MegaMoePlacement`, or None without a controller.
* `MegaMoePlacement.counters()` (every forward): the layer's `RouteCounters` for `dsv4_route_nvfp4_quant(counters=)`,
  so the routing launch counts its own picks (once the controller is finalized, for the whole run: its device flag stops
  the counting at the freeze, also inside captured CUDA graphs).
* A round moves the layer's per-slot tensors GPU to GPU in place (the controller's collective NCCL exchange; nothing is
  reallocated, so captured graphs stay valid) and calls back `on_change(new slot -> expert list)`, where the target
  rewrites its slot table in place.

The controller reads the layer's per-slot state from a view object: the target's tensors under the controller's names
for them (`SLOT_NAMES`), and every other tensor of the layer under its own name, so a per-slot tensor the mapping
misses makes the controller refuse to move the layer instead of moving it partially.
"""

from __future__ import annotations

import weakref
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm.logger import logger

#: the controller's name of each per-slot tensor -> the attribute of a target `_MegaMoe` that holds it
SLOT_NAMES = {
    "mega_fc1_weight": "fc1",
    "mega_fc1_weight_sf": "fc1_sf",
    "mega_fc2_weight": "fc2",
    "mega_fc2_weight_sf": "fc2_sf",
    "fc31_alpha": "fc1_alpha",
    "fc2_alpha": "fc2_alpha",
    "fc1_norm_const": "fc1_norm_const",
}


def megamoe_placement_controller():
    """The engine's warmup placement controller of the model being built, or None (no load balancer in this build, or
    one without `warmup_placement`)."""
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import get_moe_load_balancer

    balancer = get_moe_load_balancer()
    return getattr(balancer, "warmup_placement", None) if balancer is not None else None


class _SlotView:
    """The weights owner the controller sees: per-slot tensors by its names, the rest by the layer's own names."""

    def __init__(self, layer: torch.nn.Module, on_change: Callable[[List[int]], None]):
        self._layer = weakref.ref(layer)
        self._on_change = on_change
        self._attr_of = dict(SLOT_NAMES)

    def __getattr__(self, name: str):
        attr = self.__dict__.get("_attr_of", {}).get(name)
        if attr is None:
            raise AttributeError(name)
        return getattr(self._layer(), attr, None)

    def _tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        layer = self._layer()
        named = {attr: name for name, attr in self._attr_of.items()}
        for own, t in [*layer.named_parameters(recurse=False), *layer.named_buffers(recurse=False)]:
            yield named.get(own, own), t

    def named_parameters(self, recurse: bool = False):
        return iter(())

    def named_buffers(self, recurse: bool = False):
        return self._tensors()

    def on_expert_placement_changed(self, global_assignments: List[int]) -> None:
        self._on_change(list(global_assignments))


class _StaticTable:
    """The balancer slot of a controller layer: the target routes with its own slot table (rewritten by `on_change`),
    so the engine balancer's device table has nothing to hold."""

    def set_initial_weight_assignments(self, assignments: Sequence[int]) -> None:
        pass


class MegaMoePlacement:
    """One registered layer: its counters for the routing launch."""

    def __init__(self, controller, index: int, view: _SlotView):
        self.controller, self.index, self.view = controller, index, view

    def counters(self):
        """`RouteCounters` for `dsv4_route_nvfp4_quant(counters=)`, or None before the controller is finalized."""
        return self.controller.route_counters(self.index, 0)


def megamoe_placement_register(
    layer: torch.nn.Module,
    num_experts: int,
    top_k: int,
    slots_per_rank: int,
    assignments: Sequence[int],
    on_change: Callable[[List[int]], None],
    controller=None,
) -> Optional[MegaMoePlacement]:
    """Register `layer` (slot -> expert `assignments`, `slots_per_rank` slots per EP rank) with the engine's warmup
    placement controller (`controller`, default: this build's), or return None when there is none or it keeps the
    layer's placement. Every rank registers the same layers in the same order, before the controller is finalized.

    `on_change(new)` runs (device idle) after a round moved this layer's per-slot tensors to the placement `new`: the
    target rewrites its slot table in place and keeps `new` as its assignment."""
    from tensorrt_llm._torch.moe.fused_moe.moe_warmup_placement import _SLOT_PARAM_NAMES

    missing = sorted(set(SLOT_NAMES) - set(_SLOT_PARAM_NAMES))
    if missing:
        raise RuntimeError(
            f"the engine's warmup placement no longer moves {missing}; update SLOT_NAMES"
        )
    controller = controller if controller is not None else megamoe_placement_controller()
    if controller is None:
        return None
    view = _SlotView(layer, on_change)
    index = controller.register_layer(
        view, _StaticTable(), num_experts, top_k, slots_per_rank, list(assignments)
    )
    if index is None:
        logger.warning(
            "modeling_v2 deepseek_v41: the warmup placement keeps this MegaMoE layer's placement"
        )
        return None
    return MegaMoePlacement(controller, index, view)
