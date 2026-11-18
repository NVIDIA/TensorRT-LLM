# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel load balancing (EPLB): slot layouts, layer registration, routing and,
under online EPLB, the layer's share of moving experts between slots.

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

Online EPLB (`layer_updates_per_iter > 0`) moves experts between slots while the
engine serves: between two forwards the balancer copies an expert's weights from
host memory into a slot and points the routing at it. A target takes part in
three more steps:

4. `EplbLayer.share_experts`, while loading: host copies of this rank's share of
   the layer's experts, every per-slot weight of each in the slot layout.
5. `EplbLayer.bind_slots`, after loading and before any view of the slot weights
   is taken: move them to memory the balancer can write, and register each slot.
6. `EplbLayer.close`, after the layer's last expert GEMM of a forward: hand the
   layer back to the balancer. `route` takes it on the forward's first call and
   records which experts the tokens chose, the statistic the balancer places by.
   A target whose ranks each route their own tokens and exchange them with an
   alltoall registers the layer with `alltoall=True`, and its forward's last
   dispatch carries the statistic: `EplbLayer.local_statistic` goes into that
   dispatch and `EplbLayer.gathered` takes every rank's back.

`eplb_mode` says whether the engine runs EPLB for a deployment and in which
mode. A target lists the modes it implements in `EPLB_MODES`, and the resolver
declines EPLB deployments in any other mode (see the Modeling V2 README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import torch
from torch import nn

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2._router_index import moe_eplb_mode
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
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


#: Row ranges `(start, stop)` of a routed batch; see `EplbLayer.route`.
Rows = Sequence[tuple[int, int]]


def rows_within(
    rows: Optional[Rows], start: int, stop: int
) -> Optional[tuple[tuple[int, int], ...]]:
    """The parts of `rows` inside `[start, stop)`, relative to `start`: the token rows of
    one chunk of a batch whose token rows are `rows`. None stays None (every row a token)."""
    if rows is None:
        return None
    return tuple(
        (max(a, start) - start, min(b, stop) - start) for a, b in rows if a < stop and b > start
    )


def _token_rows(ids: torch.Tensor, rows: Optional[Rows]) -> Optional[torch.Tensor]:
    """The rows of `ids` that `rows` names, or None when it names none."""
    if rows is None:
        return ids
    parts = [ids[a:b] for a, b in rows if b > a]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else torch.cat(parts)


class EplbLayer:
    """A MoE layer registered with the engine's load balancer: its routing and, under
    online EPLB, its part in moving experts between slots.

    Built with `register` while the target is constructed, inside the model
    loader's load-balancer context, once per MoE layer and in the order the
    layers run.

    Under online EPLB the balancer takes the layers in that order on every
    forward: it hands a layer to the GPU, waits for the GPU to hand it back, and
    may then rewrite the layer's slots before the next forward. The layer is
    taken on its first `route` of a forward and handed back by `close`, so a
    target calls `close` once per forward per layer: after the layer's last
    expert GEMM, before the next layer's first `route`, and also on a forward
    that ran no tokens through the layer. A layer the forward leaves open stalls
    the balancer and, with it, the next forward.

    How the layer sees a forward's tokens follows the target's MoE round trip, chosen
    at `register`:

    * Gathered (`alltoall=False`): every rank routes the same rows, as under attention
      TP or attention DP over an allgather. Every rank records the same expert ids,
      which is the statistic of the whole batch, and picks the same slot for each of a
      token's experts, so the replica choice ignores the rank: the ranks' slot windows
      then hold each token's experts exactly once between them.
    * Alltoall (`alltoall=True`): every rank routes its own tokens and an alltoall
      carries them to the ranks that hold their slots. Every rank counts its own
      tokens' experts; the forward's last dispatch carries this rank's histogram
      (`local_statistic`) and returns every rank's, which `gathered` hands the
      balancer to sum. The replica choice is offset by the rank, as the built-in MoE's
      is under attention DP, so the ranks' tokens spread over an expert's replicas.
      Every rank routes every round of a forward, also with no tokens of its own, so
      every rank carries a histogram on the same dispatch.

    Attributes:
        balancer_layer: The layer's handle in the engine's load balancer.
        slots: The layer's slots on this rank, as loaded; under online EPLB the
            balancer moves experts between them afterwards.
        online: Whether the balancer moves experts between slots while serving.
        alltoall: Whether every rank routes its own tokens (see above).
        repeat_count: How many times one forward runs the layer.
    """

    def __init__(
        self,
        balancer_layer: SingleLayerMoeLoadBalancer,
        slots: EplbSlots,
        *,
        top_k: int = 1,
        repeat_count: int = 1,
        alltoall: bool = False,
    ):
        self.balancer_layer = balancer_layer
        self.slots = slots
        self.online = balancer_layer.is_dynamic_routing()
        self.alltoall = alltoall
        self.repeat_count = repeat_count
        self._top_k = top_k
        self._taken = False
        self._counted = False
        self._count_closed = False
        # Alltoall: route calls of this forward that ended a pass over its tokens, and
        # whether the balancer has this forward's gathered histogram.
        self._passes = 0
        self._gathered = False
        self._no_tokens: Optional[torch.Tensor] = None
        self._no_statistic: Optional[torch.Tensor] = None
        # The ids each statistic of this forward read. The balancer may count them on its
        # own stream, which joins the forward's stream only in `close`, so they must
        # outlive the caller's references until then.
        self._counted_ids: list[torch.Tensor] = []
        self._bound: Optional[dict[str, int]] = None

    @classmethod
    def register(
        cls, slots: EplbSlots, top_k: int, repeat_count: int = 1, *, alltoall: bool = False
    ) -> "EplbLayer":
        """Add the layer to the engine's load balancer with its initial placement.

        Args:
            slots: The layer's slots on this rank; its `global_assignments` are the
                initial placement.
            top_k: Experts each token selects.
            repeat_count: How many times one forward runs the layer, such as the
                draft steps of a one-model MTP layer.
            alltoall: Whether every rank routes its own tokens and an alltoall carries
                them to the slots' ranks, rather than every rank routing the same rows
                (see the class docstring).

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
        return cls(balancer_layer, slots, top_k=top_k, repeat_count=repeat_count, alltoall=alltoall)

    def route(
        self, token_selected_experts: torch.Tensor, rows: Optional[Rows] = None, last: bool = False
    ) -> torch.Tensor:
        """Global expert ids `[T, top_k]` int32 -> global slot ids `[T, top_k]` int32.

        Gathered, every rank that routes the same tokens gets the same slots
        (`offset_by_ep_rank=False`); alltoall, the replica choice is offset by the
        rank (see the class docstring).

        Under online EPLB the first call of a forward takes the layer from the
        balancer, and every call records the experts its tokens chose: the same ids
        on every rank when gathered, this rank's own when alltoall.

        Args:
            token_selected_experts: The router's expert ids.
            rows: Under online EPLB, the rows of `token_selected_experts` that are
                tokens, as `(start, stop)` ranges, when the others are padding; the
                statistic leaves padding out. None means every row is a token.
            last: Under online EPLB, no token of this pass over the layer follows
                these. Gathered, it completes the statistic here rather than in
                `close`. Alltoall, the pass that completes `repeat_count` of them
                completes this rank's histogram for the dispatch that follows
                (`local_statistic`).
        """
        if self.online:
            self._take()
            if self.alltoall:
                self._count_own(token_selected_experts, rows, last)
            else:
                ids = _token_rows(token_selected_experts, rows)
                if ids is not None:
                    self.balancer_layer.update_statistic_with_global_ids(
                        ids, not self._counted, last
                    )
                    self._remember(ids)
                    self._count_closed = last
                    self._sentinel(ids)
        if token_selected_experts.shape[0] == 0:
            # A rank with no tokens this round under an alltoall: nothing to route.
            return token_selected_experts
        return eplb_route(token_selected_experts, self.balancer_layer, self.alltoall)

    def _count_own(self, token_selected_experts: torch.Tensor, rows: Optional[Rows], last: bool):
        """Alltoall: add this rank's tokens' experts to its histogram. A call with no
        tokens of this rank still counts, an empty row: the histogram has to exist on
        every rank for the dispatch that carries it."""
        ids = _token_rows(token_selected_experts, rows)
        if ids is None or ids.numel() == 0:
            ids = self._sentinel(token_selected_experts)
        final = last and self._passes + 1 == self.repeat_count
        self.balancer_layer.update_local_statistic(ids, not self._counted, final)
        self._remember(ids)
        self._count_closed = final
        self._passes += int(last)

    def local_statistic(self) -> Optional[torch.Tensor]:
        """Alltoall, under online EPLB: this rank's histogram of the forward's experts,
        `[num_experts]`, once the `route` call that completes it has counted. The
        dispatch that follows that call carries it to every rank, and `gathered` takes
        what it returns. None on every other call, and gathered or without online EPLB.

        Every rank routes every round, so every rank gets it on the same call."""
        if not (self.online and self.alltoall and self._count_closed and not self._gathered):
            return None
        return self.balancer_layer.get_local_statistic_tensor()

    def gathered(self, statistic: torch.Tensor) -> None:
        """Alltoall: hand the balancer every rank's histogram, `[ep_size, num_experts]`, as
        the dispatch that carried `local_statistic` returns it. The balancer places by
        their sum."""
        assert self.alltoall and self._count_closed and not self._gathered, (
            "gathered() takes the histogram local_statistic() handed the dispatch, once"
        )
        self.balancer_layer.update_statistic_with_gathered_statistic(statistic)
        self._gathered = True

    def close(self) -> None:
        """Hand the layer back to the balancer for this forward, after its last expert
        GEMM; the balancer may rewrite its slots from then until the next forward takes
        it. A no-op without online EPLB.

        A forward that ran no tokens through the layer, as one that drafted nothing,
        still completes a statistic, an empty one: the balancer ages its window every
        forward, as it does for the built-in MoE's empty batches. Alltoall, a forward
        whose histogram no dispatch carried -- no rank ran the layer, or ran fewer passes
        than `repeat_count` -- completes an empty one on every rank alike, rather than
        each rank its own."""
        if not self.online:
            return
        self._take()
        if self.alltoall:
            if not self._gathered:
                if not self._count_closed:
                    self.balancer_layer.update_local_statistic(
                        self._sentinel(), not self._counted, True
                    )
                self.balancer_layer.get_local_statistic_tensor()
                self.balancer_layer.update_statistic_with_gathered_statistic(
                    self._empty_statistic()
                )
        elif not self._count_closed:
            self.balancer_layer.update_statistic_with_global_ids(
                self._sentinel(), not self._counted, True
            )
        self.balancer_layer.start_set_cpu_stage()
        self.balancer_layer.done_set_cpu_stage()
        self._taken = self._counted = self._count_closed = self._gathered = False
        self._passes = 0
        self._counted_ids.clear()

    def _remember(self, ids: torch.Tensor) -> None:
        self._counted_ids.append(ids)
        self._counted = True

    def _sentinel(self, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One row of ids no expert has: counted, it counts nothing but completes a stage.
        Made on the device and with the width of the first ids the layer routes, or of
        `top_k` on the current device when it closes a forward before routing any."""
        if self._no_tokens is None:
            if like is not None:
                width, device = like.shape[-1], like.device
            else:
                width = self._top_k
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._no_tokens = torch.full(
                (1, width), self.slots.num_experts, dtype=torch.int32, device=device
            )
        return self._no_tokens

    def _empty_statistic(self) -> torch.Tensor:
        """Every rank's histogram of a forward that ran no tokens: zeros, `[ep_size,
        num_experts]`, on the device the layer counts on."""
        if self._no_statistic is None:
            ep_size = self.slots.num_slots // self.slots.local_slots
            self._no_statistic = torch.zeros(
                (ep_size, self.slots.num_experts), dtype=torch.int32, device=self._sentinel().device
            )
        return self._no_statistic

    def _take(self) -> None:
        """Wait, once per forward, until the balancer hands the layer to the GPU."""
        if not self._taken:
            self.balancer_layer.start_wait_gpu_stage()
            self.balancer_layer.done_wait_gpu_stage()
            self._taken = True

    @property
    def host_experts(self) -> tuple[int, ...]:
        """The experts this rank keeps host copies of for the balancer under online EPLB:
        its share of the ranks on its node, so the node holds every expert once. Empty
        without online EPLB."""
        if not self.online:
            return ()
        return tuple(self.balancer_layer.get_load_expert_ids())

    def share_experts(
        self,
        slot_weights: Mapping[str, torch.Tensor],
        expert_weight: Callable[[str, int], torch.Tensor],
    ) -> None:
        """Hand the balancer this rank's host copies of the layer's experts, under online
        EPLB, while loading. A no-op otherwise.

        Args:
            slot_weights: Every per-slot weight of the layer by name, `[local_slots, ...]`;
                an expert's copy has the shape and dtype of one slot.
            expert_weight: `(name, expert) -> tensor`: expert `expert`'s `name` weight in
                the slot layout, on any device. Called for the experts in `host_experts`.

        The copies move to memory the ranks of the node share before this returns, so
        the private ones go with each layer rather than piling up over the model.
        """
        if not self.online:
            return
        sharer = self.balancer_layer.host_tensor_sharer
        mine = set(self.host_experts)
        for name, slot_weight in slot_weights.items():
            shape, dtype = slot_weight.shape[1:], slot_weight.dtype
            for expert in range(self.slots.num_experts):
                if expert not in mine:
                    sharer.pre_register_host_tensor_with_shape(expert, name, dtype, shape)
                    continue
                t = expert_weight(name, expert).to("cpu").contiguous()
                assert t.shape == shape and t.dtype == dtype, (
                    f"expert {expert}'s {name} is {tuple(t.shape)} {t.dtype}, a slot is "
                    f"{tuple(shape)} {dtype}"
                )
                sharer.share_host_tensor_with_shape(expert, name, t)
        sharer.finalize_layer_weights()

    def bind_slots(self, slot_weights: Mapping[str, torch.Tensor]) -> None:
        """Move the layer's per-slot weights to memory the balancer can write and register
        every slot of each, under online EPLB, after loading. A no-op otherwise.

        Args:
            slot_weights: Every per-slot weight of the layer by name, `[local_slots, ...]`
                and contiguous -- the names `share_experts` was given. A parameter
                keeps its identity and takes the new storage as its data.

        A view taken before this call keeps reading the old storage, which the
        balancer never writes: bind first, then take the views the forward reads.
        The slots stay registered: binding the same weights again does nothing, as
        when the model loader runs a target's post-load hook more than once.
        """
        if not self.online:
            return
        if self._bound is not None:
            assert {name: w.data_ptr() for name, w in slot_weights.items()} == self._bound, (
                "the layer's slot weights moved after the balancer registered them"
            )
            return
        for name, weight in slot_weights.items():
            data = weight.detach() if isinstance(weight, nn.Parameter) else weight
            assert data.is_contiguous() and data.shape[0] == self.slots.local_slots, (
                f"{name} {tuple(data.shape)} is not {self.slots.local_slots} contiguous slots"
            )
            self.balancer_layer.make_tensor_host_accessible(data)
            if isinstance(weight, nn.Parameter):
                weight.data = data
            for local in range(self.slots.local_slots):
                self.balancer_layer.register_weight_slot(local, name, data[local])
        self._bound = {name: w.data_ptr() for name, w in slot_weights.items()}
