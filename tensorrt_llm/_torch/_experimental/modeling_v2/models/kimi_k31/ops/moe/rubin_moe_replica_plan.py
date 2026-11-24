# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer dynamic EPLB of the Rubin CuteDSL context MoE (the routed experts behind the one-sided
all-to-all): the replica plan of every routed call.

The engine's ``per_layer_dynamic_eplb`` manager (``moe_config.load_balancer`` ``mode:
per_layer_dynamic``) owns the planner of every MoE layer. A layer keeps its ``H = experts / EP``
resident experts and gains ``S = (num_slots - experts) / EP`` helper slots per rank; the routed id
space becomes the physical slots ``rank * (H + S) + slot`` (``num_slots`` ids, rank ``slot // (H +
S)``). Per routed call, ``rubin_moe_replica_route`` submits the layer's HALO-Q + copy kernel on
the manager's copy stream (it exchanges every rank's routing counts, picks the hot experts, writes
each pick's physical slot and copies the hot experts' weights into the helper slots of their
ranks); the returned plan orders the first reader of the physical slots (``wait_for_routes``,
before the dispatch) and the first reader of the helper slots (``wait_for_helper_weights``,
before the experts), and the layer releases it once the experts' kernels are queued
(``finish_replica_plan``).

Runtime helpers (builders, run once): ``replica_layers`` registers a target's MoE layers with the
manager at construction (collective-free); ``bind_replica_planes`` moves one layer's per-expert
kernel operands (``LIVE_PLANES``: the NVFP4 FC1 / FC2 weights, their swizzled block scales and
the FP32 alphas, ``[H, ...]`` each) onto its ``H + S`` live-weight arena and builds the layer's
producer, after loading (collective over the EP group, every rank the same layers in the same
order). ``RESIDENT_PLAN`` is the plan of a layer without a load balancer: its waits are free.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: A layer's per-expert kernel operands that move onto its live-weight arena, in arena order: the
#: FC1 rows (16 gate then 16 up) and FC2 weights (E2M1 pairs), their 128x4-swizzled E4M3 block
#: scales and the FP32 FC1 / FC2 dequantization scales (every EP rank uses the same order).
LIVE_PLANES = ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "fc31_alpha", "fc2_alpha")


def _no_wait() -> None:
    return None


class _ResidentPlan:
    """The plan of a routed call without a load balancer: the routes are the logical experts, the
    experts read only resident weights, so neither wait orders anything."""

    wait_for_routes = staticmethod(_no_wait)
    wait_for_helper_weights = staticmethod(_no_wait)


RESIDENT_PLAN = _ResidentPlan()


class _PlaneOwner:
    """What ``bind_live_weight_planes`` and the layer's producer read of a backend: the planes by
    name, the layer handle, the resident count, the mapping, the layer index, the EP process group
    (None: the manager forms an MPI communicator of ``mapping``'s EP group) and the most rows one
    rank routes in one call (the planner's route capacity)."""

    ep_process_group = None
    tactic_autotune = False

    def __init__(
        self,
        handle: Any,
        planes: Mapping[str, torch.Tensor],
        mapping: Any,
        layer_idx: int,
        max_tokens_per_rank: int,
    ) -> None:
        self.layer_load_balancer = handle
        self.expert_size_per_partition = int(handle.resident_experts_per_rank)
        self.mapping = mapping
        self.layer_idx = int(layer_idx)
        self.max_tokens_per_rank_capacity = int(max_tokens_per_rank)
        for name in LIVE_PLANES:
            setattr(self, name, planes[name])


def current_replica_manager() -> Optional[Any]:
    """The engine's MoE load balancer current while the model is built (the model loader enters
    it around construction when ``moe_config.load_balancer`` is set), or None."""
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import get_moe_load_balancer

    return get_moe_load_balancer()


def replica_layers(manager: Any, num_experts: int, top_k: int, count: int) -> tuple:
    """Register ``count`` MoE layers of ``num_experts`` routed experts (top ``top_k``) with the
    per-layer dynamic EPLB ``manager`` (the engine's ``get_moe_load_balancer()`` while the model is
    built), in layer order; return their handles. Collective-free."""
    if not callable(getattr(manager, "add_layer", None)) or not hasattr(manager, "num_local_slots"):
        raise TypeError(
            f"the Rubin CuteDSL context MoE takes per-layer dynamic EPLB only, got {manager!r}"
        )
    handles = tuple(
        manager.add_layer(num_experts, top_k, manager.num_local_slots) for _ in range(count)
    )
    for handle in handles:
        if not handle.produces_replica_plan:
            raise TypeError(f"{handle!r} produces no replica plan")
    return handles


def bind_replica_planes(
    handle: Any,
    planes: Mapping[str, torch.Tensor],
    *,
    mapping: Any,
    layer_idx: int,
    max_tokens_per_rank: int,
    hidden: int,
    intermediate: int,
) -> _PlaneOwner:
    """Move ``planes`` (``LIVE_PLANES`` -> ``[H, ...]`` CUDA tensors, nn.Parameters are kept and
    re-pointed) onto ``handle``'s ``H + S`` live-weight arena and build the layer's HALO-Q / copy
    producer. Collective over the EP group. Returns the owner the producer was built for; the
    caller keeps it alive with the layer."""
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.live_planes import (
        bind_live_weight_planes,
    )

    owner = _PlaneOwner(handle, planes, mapping, layer_idx, max_tokens_per_rank)
    bind_live_weight_planes(
        owner, LIVE_PLANES, hidden=hidden, intermediate=intermediate, ep_process_group=None
    )
    # The replaced [H, ...] tensors are free: return them before the next layer's arena.
    torch.cuda.empty_cache()
    handle.bind_backend(owner)
    return owner


class _CudaView:
    """``__cuda_array_interface__`` of a fabric mapping (a peer's or this rank's buffer)."""

    def __init__(self, pointer: int, shape: tuple) -> None:
        self.__cuda_array_interface__ = {
            "data": (int(pointer), False),
            "shape": tuple(shape),
            "typestr": "<i4",
            "strides": None,
            "version": 3,
        }


class ReplicaCadence:
    """When a forward replans: every ``every``-th forward runs the layers' HALO-Q plans (and their
    helper copies); the forwards between reuse each layer's last helper placement. Every EP rank
    counts the same forwards (the routed exchange is collective). ``begin`` starts a forward."""

    def __init__(self, every: int) -> None:
        if int(every) < 1:
            raise ValueError(f"the replica plan cadence must be at least 1, got {every}")
        self.every = int(every)
        self.forwards = 0
        self.replan = True

    def begin(self) -> None:
        self.replan = self.forwards % self.every == 0
        self.forwards += 1


class ReplicaPlacements:
    """Every EP rank's helper placement of every EPLB layer, ``[EP, layers, S]`` int32 expert ids
    in fabric memory: a rank writes its row of a layer after the layer's HALO-Q plan (the
    scheduler's ``hot_expert_ids``) and reads every rank's row when it reuses the placement.
    Collective at construction (fabric handle exchange over the manager's EP communicator)."""

    def __init__(self, manager: Any, mapping: Any, layers: int, helpers: int) -> None:
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler.multirank_dist import (  # noqa: E501
            FabricSymmetricBuffer,
        )

        comm = manager.get_ep_comm(None, mapping)
        device = torch.cuda.current_device()
        self.shape = (int(layers), int(helpers))
        self.buffer = FabricSymmetricBuffer(4 * layers * helpers, device)
        handles = list(comm.allgather(self.buffer.shareable))
        rank = int(mapping.moe_ep_rank)
        self.rows = [
            torch.as_tensor(
                _CudaView(self.buffer.ptr if r == rank else self.buffer.import_peer(h), self.shape),
                device=f"cuda:{device}",
            )
            for r, h in enumerate(handles)
        ]
        self.mine = self.rows[rank]
        self.mine.fill_(-1)
        torch.cuda.synchronize()
        comm.barrier()


class ReplicaRouter:
    """One layer's routing under per-layer dynamic EPLB with a plan cadence: on a replanning
    forward the layer's HALO-Q plan (``rubin_moe_replica_route``) and its helper placement
    published to ``placements``; between them the routes of the last placement (no exchange,
    no copy; the helper rows hold their experts, ``helper_slot_mode: per_layer_resident``)."""

    def __init__(self, handle: Any, cadence: ReplicaCadence, placements: Any, row: int) -> None:
        self.handle, self.cadence, self.placements, self.row = handle, cadence, placements, row
        self.home = int(handle.resident_experts_per_rank)
        self.slots = int(handle.compute_slots_per_rank)
        self.experts = int(handle.expert_count)

    def plan(self, ids: torch.Tensor) -> tuple:
        if self.cadence.replan:
            slots, plan = rubin_moe_replica_route(self.handle, ids)
            group = self.handle._scheduler_group
            # On the copy stream, after the plan wrote this rank's placement.
            with torch.cuda.stream(group.copy_stream):
                self.placements.mine[self.row].copy_(group.scheduler.outputs.hot_expert_ids)
            return slots, plan
        return reuse_routes(ids, self.placements, self.row, self.home, self.slots, self.experts), (
            RESIDENT_PLAN
        )

    def finish(self, plan: Any) -> None:
        if plan is not RESIDENT_PLAN:
            self.handle.finish_replica_plan(plan)


def reuse_routes(
    ids: torch.Tensor, placements: Any, row: int, home: int, slots: int, experts: int
) -> torch.Tensor:
    """Physical slots of ``ids`` (int32 ``[rows, top_k]`` logical experts) on the last helper
    placement: each pick of an expert with replicas goes to its resident slot or one of its
    helper slots, chosen by the token's row (an even split of the expert's rows)."""
    helpers = torch.stack([r[row] for r in placements.rows]).long()  # [EP, S]
    ep, s = helpers.shape
    slot_of = (
        torch.arange(ep, device=ids.device).unsqueeze(1) * slots
        + home
        + torch.arange(s, device=ids.device).unsqueeze(0)
    ).reshape(-1)
    held = helpers.reshape(-1)
    valid = held >= 0
    held, slot_of = held[valid], slot_of[valid]
    order = torch.argsort(held, stable=True)
    held, slot_of = held[order], slot_of[order]
    counts = torch.bincount(held, minlength=experts)
    start = torch.cumsum(counts, 0) - counts
    rank_in = torch.arange(held.numel(), device=ids.device) - start[held]
    width = max(int(s * ep), 1)
    table = torch.zeros(experts, width, dtype=torch.long, device=ids.device)
    table[held, rank_in] = slot_of
    e = ids.long()
    resident = (e // home) * slots + e % home
    n = counts[e] + 1
    token = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1)
    j = token % n
    replica = table[e, (j - 1).clamp_min(0)]
    return torch.where(j == 0, resident, replica).to(torch.int32)


class RubinMoeReplicaRoute(OpWrapper):
    """Submit one routed call's replica plan: ``token_selected_experts`` (int32 ``[rows, top_k]``,
    dense, logical expert ids) to ``layer`` (a bound ``PerLayerDynamicMoeLayerLoadBalancer``).
    Returns ``(slots, plan)``: the physical slots (int32 ``[rows, top_k]``, written on the copy
    stream; read them after ``plan.wait_for_routes()``) and the plan, which ``layer`` releases in
    ``finish_replica_plan`` once the experts' kernels are queued. Collective over the EP group
    (every rank calls it for the same layers in the same order); no host wait."""

    ARCHS = frozenset({Arch.SM_107})

    note = """
    A pick whose physical slot is a helper slot is served by a byte-exact copy of its expert, so
    the layer's output equals the output without a plan up to the order of the finalize's adds.
    The resident slots of rank r hold its experts r * H .. r * H + H - 1 at slots 0 .. H - 1.
    """

    def __call__(self, layer: Any, token_selected_experts: torch.Tensor) -> tuple:
        slots = layer.route(token_selected_experts)
        return slots, layer.require_replica_plan()

    def reference(self, layer: Any, token_selected_experts: torch.Tensor) -> torch.Tensor:
        """The physical slots of a plan that replicates nothing: each expert's resident slot."""
        home = int(layer.resident_experts_per_rank)
        slots = int(layer.compute_slots_per_rank)
        ids = token_selected_experts
        return (ids // home) * slots + ids % home

    def is_valid(self, layer: Any, token_selected_experts: torch.Tensor) -> None:
        ids = token_selected_experts
        assert ids.dtype == torch.int32, "the planner reads int32 expert ids"
        assert ids.dim() == 2 and ids.shape[1] == layer.top_k, "one row of top_k ids per token"
        assert ids.is_contiguous(), "the planner reads the ids' data_ptr()"
        assert layer.replica_plan is None, "the layer's previous plan is not finished"


rubin_moe_replica_route = RubinMoeReplicaRoute()


def plane_tensors(params: Mapping[str, torch.Tensor], prefix: str) -> dict:
    """``LIVE_PLANES`` -> ``params[prefix + name]``."""
    return {name: params[prefix + name] for name in LIVE_PLANES}
