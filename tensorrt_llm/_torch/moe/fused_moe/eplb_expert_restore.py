# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Restore a failed rank's experts into EPLB spare slots under Wide-EP fault tolerance.

Every node keeps host weights for all experts. After a failure, every active rank plans the same
placement and loads the experts it receives from its node's host weights.
"""

import collections
import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from tensorrt_llm._wide_ep_fault_tolerance import TURN_FAULT_TOLERANCE_OFF
from tensorrt_llm.bindings.internal import runtime
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from .moe_load_balancer import HostMoeTensorSharer, MoeLoadBalancer, SingleLayerMoeLoadBalancer


def plan_layer_placement(
    current: np.ndarray,
    base: np.ndarray,
    active: np.ndarray,
    active_slot_count: int,
    expert_count: int,
    layer_idx: int,
) -> np.ndarray:
    """Return a deterministic ``[ranks, slots]`` placement of every expert on the ``active`` ranks.

    Active ranks keep their active slots from ``base`` and their spare slots from ``current``. An
    active slot that repeats an expert of an earlier active slot holds an extra replica, which an
    interrupted online EPLB update can leave behind. A lost expert takes an emptied active slot
    first. Next it takes an extra replica's slot if routing uses one replica per expert, else an
    empty spare slot, and an extra replica's slot only when no spare slot is left.

    Args:
        current: ``[ranks, slots]`` experts, -1 for an empty slot.
        base: ``current``, or the initial placement when a rank rejoins.
        active: ``[ranks]`` flags telling whether a rank is active.
        active_slot_count: The active slots per rank, the slots before the spare slots.
        expert_count: The number of experts of the layer.
        layer_idx: The layer index, which rotates the rank order so restored experts spread out.

    Raises:
        ValueError: The active ranks' spare slots and extra replicas cannot hold every expert.
    """
    placement = np.full_like(current, -1)
    placement[active, :active_slot_count] = base[active, :active_slot_count]
    active_ranks = np.flatnonzero(active)
    # The empty active slots and the active slots of extra replicas, in rank and slot order.
    empty_slots, extra_slots = collections.deque(), collections.deque()
    kept = np.zeros(expert_count, dtype=bool)
    for rank in active_ranks:
        for slot in range(active_slot_count):
            expert_id = placement[rank, slot]
            if expert_id < 0:
                empty_slots.append((rank, slot))
            elif kept[expert_id]:
                extra_slots.append((rank, slot))
            else:
                kept[expert_id] = True
    single_replica = active_slot_count * current.shape[0] == expert_count
    if single_replica:
        for rank, slot in extra_slots:
            placement[rank, slot] = -1
    replicas = np.bincount(placement[placement >= 0], minlength=expert_count)
    spare_slots = range(active_slot_count, current.shape[1])
    for rank in active_ranks:
        for slot in spare_slots:
            expert_id = current[rank, slot]
            if expert_id >= 0 and replicas[expert_id] == 0:
                placement[rank, slot] = expert_id
                replicas[expert_id] = 1
    rank_order = np.roll(active_ranks, -layer_idx)
    for expert_id in np.flatnonzero(replicas == 0):
        replicas[expert_id] = 1
        if empty_slots:
            placement[empty_slots.popleft()] = expert_id
            continue
        # Prefer the fewest filled spare slots, then the rank order, then a slot that held an expert
        # in ``current`` (its memory may still be backed), then the lowest slot.
        candidates = [
            (
                np.count_nonzero(placement[rank, active_slot_count:] >= 0),
                order,
                current[rank, slot] < 0,
                slot,
            )
            for order, rank in enumerate(rank_order)
            for slot in spare_slots
            if placement[rank, slot] < 0
        ]
        if extra_slots and (single_replica or not candidates):
            placement[extra_slots.popleft()] = expert_id
        elif candidates:
            _, order, _, slot = min(candidates)
            placement[rank_order[order], slot] = expert_id
        else:
            raise ValueError(
                f"The spare slots of active ranks {active_ranks.tolist()} cannot hold expert "
                f"{expert_id}"
            )
    return placement


@dataclass(frozen=True, eq=False)
class ExpertRestorePlan:
    """Per-layer placements for one set of active ranks.

    Every field except ``device_bytes`` is the same on every active rank.
    """

    active_ranks: tuple[bool, ...]
    placements: tuple[np.ndarray, ...]
    placement_before: tuple[np.ndarray, ...]
    digest: str
    device_bytes: int


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Return a flat uint8 view of the contiguous ``tensor``."""
    return tensor.view(-1).view(torch.uint8)


def _phase_done(times: Optional[dict[str, float]], phase: str, start_s: float) -> float:
    """Record the milliseconds since ``start_s`` as ``phase`` in ``times`` and return the time."""
    now_s = time.perf_counter()
    if times is not None:
        times[phase] = (now_s - start_s) * 1e3
    return now_s


def create_spare_slot_storage(
    layer: SingleLayerMoeLoadBalancer, name: str, slots: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return storage for ``name`` whose spare slots map no memory, or None if that saves none."""
    slot_bytes = slots.numel() // slots.shape[0] * slots.element_size()
    spare_slots = list(layer.spare_slots)
    if not torch.ops.trtllm.moe_unbacked_slots_save_memory(slot_bytes, slots.shape[0], spare_slots):
        return None
    storage = torch.ops.trtllm.moe_unbacked_slot_tensor(
        list(slots.shape), slots.dtype, spare_slots, layer.updates_enabled
    )
    if slots.device.type != "meta":
        active = layer.active_slot_count_per_rank
        storage[:active].copy_(slots[:active])
    layer.unbacked_slot_tensors[name] = storage
    return storage


def slot_parameters(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the named weights of ``module`` that have one row per local slot.

    Restoring an expert into a spare slot needs host weights for each of them.
    ``register_weight_slots_after_to_cuda`` checks this.
    """
    return [
        (name, param)
        for name, param in module.named_parameters(recurse=False)
        if param.dim() > 0 and param.shape[0] == module.expert_size_per_partition
    ]


def create_spare_slot_weights(module: torch.nn.Module) -> None:
    """Move the slot weights of ``module`` to unbacked spare slots."""
    layer = module.layer_load_balancer
    for name, param in slot_parameters(module):
        storage = create_spare_slot_storage(layer, name, param)
        if storage is not None:
            module.register_parameter(
                name, torch.nn.Parameter(storage, requires_grad=param.requires_grad)
            )


def use_spare_slot_storage(layer: SingleLayerMoeLoadBalancer, wt: torch.Tensor, name: str) -> None:
    """Take the loaded ``wt`` as the slots of weight ``name``."""
    unbacked = layer.unbacked_slot_tensors.get(name)
    if unbacked is None:
        if layer.updates_enabled:
            layer.make_tensor_host_accessible(wt)
    elif wt.untyped_storage().data_ptr() != unbacked.untyped_storage().data_ptr():
        raise RuntimeError(
            f"Weight {name} of MoE layer {layer.get_layer_idx()} lost the storage whose spare "
            "slots map no device memory: a transform after loading replaced it"
        )
    layer.slot_tensors[name] = wt


def filled_spare_slots(layer: SingleLayerMoeLoadBalancer, local: np.ndarray) -> list[int]:
    """Return the spare slots of ``layer`` that ``local`` fills."""
    return [slot for slot in layer.spare_slots if local[slot] >= 0]


def spare_slot_bytes_to_back(layer: SingleLayerMoeLoadBalancer, slots: Sequence[int]) -> int:
    """Return the device bytes backing ``slots`` of ``layer`` would map now."""
    if not slots:
        return 0
    return sum(
        torch.ops.trtllm.moe_unbacked_slots_bytes_to_back(tensor, list(slots))
        for tensor in layer.unbacked_slot_tensors.values()
    )


class DeploymentHostWeights(HostMoeTensorSharer):
    """Host tensor sharer that keeps the host weights on this node, which restores load slots from."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The host weights on this node by expert and weight name, which restores load slots from.
        self.node_host_weights: dict[tuple[int, str], torch.Tensor] = {}

    def finalize_host_tensor_sharing(self, add_host_weight_fn: Optional[Callable] = None):
        def keep(expert_id: int, name: str, weight: torch.Tensor) -> None:
            self.node_host_weights[(expert_id, name)] = weight
            if add_host_weight_fn is not None:
                add_host_weight_fn(expert_id, name, weight)

        super().finalize_host_tensor_sharing(keep)

    def pre_shutdown_cleanup(self):
        # Shared memory closes only once no view of it is left.
        self.node_host_weights.clear()
        super().pre_shutdown_cleanup()


class EplbExpertRestore:
    """Size the spare slots of a ``MoeLoadBalancer`` and restore failed ranks' experts into them."""

    def __init__(self, balancer: "MoeLoadBalancer"):
        self.balancer = balancer
        nodes = self._rank_nodes()
        self.largest_failure = max(map(len, self._recoverable_failures(nodes)), default=0)
        if (
            self.largest_failure
            and balancer.layer_updates_per_iter > 0
            and not runtime.host_accessible_device_memory_uses_gdrcopy()
        ):
            raise ValueError(
                "Wide-EP fault tolerance with online EPLB requires GDRCopy in CDMM mode, "
                "because online EPLB's CPU copies reach the spare slots only through GDRCopy. Set "
                f"layer_updates_per_iter=0 or {TURN_FAULT_TOLERANCE_OFF}."
            )
        self.active_ranks: tuple[bool, ...] = (True,) * balancer.ep_size
        self.backed_spare_slot_bytes = 0
        # By layer, the slots of this rank that a restore wrote and no installed placement names yet.
        self._invalid_slots: dict[int, set[int]] = {}

    def spare_slot_count(self, expert_count: int, eplb_slot_count: int) -> int:
        """Return the spare slots per rank that hold the largest recoverable failure's experts."""
        lost_experts = min(self.largest_failure * eplb_slot_count, expert_count)
        active_ranks = self.balancer.ep_size - self.largest_failure
        return ceil_div(lost_experts, active_ranks)

    # EP rank r is global rank r because fault tolerance requires pipeline_parallel_size=1.
    def _rank_nodes(self) -> list[int]:
        return [self.balancer.mapping.get_node_rank(rank) for rank in range(self.balancer.ep_size)]

    @staticmethod
    def _recoverable_failures(nodes: list[int]) -> list[list[int]]:
        """Return the lost ranks of each recoverable failure.

        A recoverable failure loses a node other than rank 0's, or one rank on rank 0's node.
        """
        node_ranks: dict[int, list[int]] = {}
        for rank, node in enumerate(nodes):
            node_ranks.setdefault(node, []).append(rank)
        rank_0_node = node_ranks.pop(nodes[0], [])
        return list(node_ranks.values()) + [[rank] for rank in rank_0_node[1:]]

    def current_placement(self) -> list[tuple[int, list[list[int]]]]:
        """Return each layer's expert count and placement as this rank sees it."""
        return [
            (layer.expert_count, self._placement_view(i, layer).tolist())
            for i, layer in enumerate(self.balancer.single_layer_load_balancers)
        ]

    def _placement_view(self, layer_idx: int, layer: SingleLayerMoeLoadBalancer) -> np.ndarray:
        """Return ``layer``'s placement as this rank sees it, with no expert in invalid slots.

        A restore that stopped before it installed its placement may have written part of a slot.
        """
        placement = np.array(layer.get_old_rank_expert_ids(), dtype=np.int32)
        placement[self.balancer.ep_rank, sorted(self._invalid_slots.get(layer_idx, ()))] = -1
        return placement

    def plan_expert_restore(
        self,
        active_ranks: Sequence[bool],
        current: Sequence[np.ndarray],
        rejoining: Optional[Sequence[bool]] = None,
    ) -> ExpertRestorePlan:
        """Plan every layer's placement for ``active_ranks``.

        Args:
            active_ranks: Whether each EP rank is active.
            current: The per-layer placement that the active ranks agreed on, so that every
                active rank plans the same placement. An active rank's row is its row of
                ``current_placement``, so a slot that a stopped restore wrote loads again.
            rejoining: Whether each EP rank rejoins. If one does, the active slots of all active
                ranks take the initial placement, which the rejoining rank loaded.

        Raises:
            RuntimeError: An iteration is active.
            ValueError: The active ranks cannot hold every expert, or a slot would need to load an
                expert whose host weights no active rank's node keeps.
        """
        balancer = self.balancer
        active = np.asarray(active_ranks, dtype=bool)
        if active.shape != (balancer.ep_size,) or not active[balancer.ep_rank]:
            raise ValueError(f"EP rank {balancer.ep_rank} plans for active ranks {active.tolist()}")
        if balancer.in_iter:
            raise RuntimeError("Cannot plan an expert restore while an iteration is active")
        rejoining_ranks = (
            np.zeros(balancer.ep_size, dtype=bool)
            if rejoining is None
            else np.asarray(rejoining, dtype=bool) & active
        )
        digest = hashlib.sha256(active.tobytes())
        placements = []
        placements_before = []
        device_bytes = 0
        for i, (layer, layer_current) in enumerate(
            zip(balancer.single_layer_load_balancers, current, strict=True)
        ):
            layer_current = np.asarray(layer_current, dtype=np.int32)
            base = (
                layer.initial_placement.reshape(layer_current.shape)
                if rejoining_ranks.any()
                else layer_current
            )
            placement = plan_layer_placement(
                layer_current,
                base,
                active,
                layer.active_slot_count_per_rank,
                layer.expert_count,
                i,
            )
            # A rejoining rank loaded its initial-placement slots from storage.
            before = np.where(rejoining_ranks[:, None], base, layer_current)
            loaded = (placement >= 0) & (placement != before)
            if loaded.any() and not layer.need_load_shared_weights():
                raise ValueError(
                    f"Layer {layer.get_layer_idx()} has no host weights to load experts "
                    f"{sorted(set(placement[loaded].tolist()))} from"
                )
            device_bytes += spare_slot_bytes_to_back(
                layer, filled_spare_slots(layer, placement[balancer.ep_rank])
            )
            digest.update(placement.tobytes())
            placements.append(placement)
            placements_before.append(before)
        return ExpertRestorePlan(
            tuple(bool(rank) for rank in active),
            tuple(placements),
            tuple(placements_before),
            digest.hexdigest()[:16],
            device_bytes,
        )

    def set_spare_slots_backed(self, layer_slots: Sequence[Sequence[int]], backed: bool) -> int:
        """Back spare slots with device memory or map them back to the shared scratch.

        Returns:
            The number of bytes whose backing changed.
        """
        tensors, slots, slot_counts = [], [], []
        for layer, spare_slots in zip(
            self.balancer.single_layer_load_balancers, layer_slots, strict=True
        ):
            if not spare_slots:
                continue
            for tensor in layer.unbacked_slot_tensors.values():
                tensors.append(tensor)
                slots.extend(spare_slots)
                slot_counts.append(len(spare_slots))
        if not tensors:
            return 0
        changed = torch.ops.trtllm.moe_unbacked_slots_set_backed(
            tensors, slots, slot_counts, backed
        )
        self.backed_spare_slot_bytes += changed if backed else -changed
        return changed

    def release_empty_spare_slots(self) -> int:
        """Map the spare slots that the installed placement leaves empty back to the shared scratch.

        An online EPLB update may still run, because it never fills these slots. An active slot
        that a stopped restore emptied is unusable too, but it has no unbacked memory to release.

        Returns:
            The number of bytes released.
        """
        balancer = self.balancer
        if balancer.in_iter:
            raise RuntimeError("Cannot release spare slots while an iteration is active")
        return self.set_spare_slots_backed(
            [
                [slot for slot in layer.get_unusable_local_slots() if slot in layer.spare_slots]
                for layer in balancer.single_layer_load_balancers
            ],
            False,
        )

    def restore_lost_experts(
        self,
        plan: ExpertRestorePlan,
        times: Optional[dict[str, float]] = None,
        progress: Callable[[], None] = lambda: None,
    ) -> int:
        """Load the slots that ``plan`` changes on this rank, install it and return the bytes mapped.

        Every active rank must restore the same plan before any rank routes with it. A slot loads
        from the host weights on this node.

        Args:
            plan: The plan that every active rank restores.
            times: Receives each phase's duration in milliseconds.
            progress: Called after the backing and each layer loaded, so that a long restore keeps
                reporting progress. It may raise to stop the restore. The slots that ``plan`` loads
                on this rank then hold no expert in ``current_placement`` until a restore installs a
                placement.

        Raises:
            RuntimeError: An iteration is active, or the online EPLB worker may still write the
                slots.
        """
        balancer = self.balancer
        if balancer.in_iter:
            raise RuntimeError("Cannot restore experts while an iteration is active")
        # No EPLB weight update may write the slots that the restore loads, or swap the placement
        # that the check below reads.
        balancer.load_balancer_impl.check_slots_writable()
        layers = balancer.single_layer_load_balancers
        if (
            not self._invalid_slots
            and plan.active_ranks == self.active_ranks
            and all(
                np.array_equal(placement, layer.get_old_rank_expert_ids())
                for placement, layer in zip(plan.placements, layers)
            )
        ):
            return 0
        # The slots that the restore loads hold no expert in this rank's view until it installs.
        for i in range(len(layers)):
            loaded = np.flatnonzero(self._loaded_slots(plan, i)).tolist()
            if loaded:
                self._invalid_slots.setdefault(i, set()).update(loaded)
        start_s = time.perf_counter()
        # A spare slot is backed before an expert is loaded into it and before routing can reach it.
        mapped = self.set_spare_slots_backed(
            [
                filled_spare_slots(layer, placement[balancer.ep_rank])
                for layer, placement in zip(layers, plan.placements)
            ],
            True,
        )
        start_s = _phase_done(times, "back", start_s)
        progress()
        self._load_local_slots(plan, progress)
        start_s = _phase_done(times, "load", start_s)
        balancer.load_balancer_impl.restore_placement(
            [rank for rank, active in enumerate(plan.active_ranks) if not active],
            [placement.tolist() for placement in plan.placements],
        )
        _phase_done(times, "install", start_s)
        self._invalid_slots.clear()
        self.active_ranks = plan.active_ranks
        return mapped

    def _loaded_slots(self, plan: ExpertRestorePlan, layer_idx: int) -> np.ndarray:
        """Return whether ``plan`` loads each slot of this rank in layer ``layer_idx``."""
        ep_rank = self.balancer.ep_rank
        placement = plan.placements[layer_idx][ep_rank]
        return (placement >= 0) & (placement != plan.placement_before[layer_idx][ep_rank])

    def _load_local_slots(self, plan: ExpertRestorePlan, progress: Callable[[], None]) -> None:
        """Load this rank's slots that ``plan`` changes from this node's host weights."""
        ep_rank = self.balancer.ep_rank
        for i, (layer, placement) in enumerate(
            zip(self.balancer.single_layer_load_balancers, plan.placements)
        ):
            changed = self._loaded_slots(plan, i)
            for slot in np.flatnonzero(changed):
                expert = int(placement[ep_rank, slot])
                for name, tensor in layer.slot_tensors.items():
                    host = layer.host_tensor_sharer.node_host_weights[(expert, name)]
                    _bytes(tensor[slot]).copy_(_bytes(host))
            progress()


class FaultTolerantMoeLoadBalancer(MoeLoadBalancer):
    """MoE load balancer whose spare slots take the experts of failed ranks."""

    fault_tolerance = True

    def __init__(self, *args, mapping: Mapping, **kwargs):
        """Create the load balancer of one rank of a deployment.

        Args:
            mapping: The mapping whose nodes decide which failures are recoverable.
        """
        super().__init__(*args, **kwargs)
        self.mapping = mapping
        self.expert_restore = EplbExpertRestore(self)
        self.host_tensor_sharer_cls = DeploymentHostWeights

    def _spare_slot_count(self, expert_count: int, slot_count_per_rank: int) -> int:
        return self.expert_restore.spare_slot_count(expert_count, slot_count_per_rank)

    @property
    def adds_spare_slots(self) -> bool:
        # A layer gets spare slots exactly when some failure is recoverable.
        return self.expert_restore.largest_failure > 0

    def finalize_model(self):
        super().finalize_model()
        spare_slot_count = max(
            (layer.spare_slot_count for layer in self.single_layer_load_balancers), default=0
        )
        if spare_slot_count and self.ep_rank == 0:
            logger.info(
                f"Wide-EP fault tolerance adds up to {spare_slot_count} unbacked spare slots per "
                "rank"
            )

    def shutdown(self, local_safe: bool = True) -> bool:
        """Release only this rank's resources. Other ranks on its node may have failed."""
        if self.is_shutdown:
            return True
        if not local_safe or self.in_iter:
            return False
        # The C++ shutdown waits for the weight updates in flight. Those read the host weights.
        self.load_balancer_impl.shutdown()
        for layer in self.single_layer_load_balancers:
            layer.py_pre_shutdown_cleanup()
        for layer in self.single_layer_load_balancers:
            layer.py_post_shutdown_cleanup()
        self.is_shutdown = True
        return True

    def abandon_iterations(self) -> None:
        """End the online EPLB work for the iterations that a rank failure interrupted.

        Every launched kernel must be able to finish, and no forward may launch until this returns.
        """
        if self.in_iter:
            raise RuntimeError("Cannot abandon iterations while an iteration is active")
        self.load_balancer_impl.abandon_iterations()
        for layer in self.single_layer_load_balancers:
            layer.abandon_forward()
