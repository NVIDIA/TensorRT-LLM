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

With static EPLB, each node keeps in /dev/shm the host weights for the next node's experts, so a
whole-node failure leaves a host copy of every lost expert. With online EPLB, every node keeps
host weights for all experts. After a failure, every active rank plans the same placement and
loads the experts it receives from its node's host weights, or through the idle MoE all-to-all
workspace from a rank whose node has them.
"""

import collections
import ctypes
import functools
import hashlib
import itertools
import os
import stat
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Any, NamedTuple, Optional

import numpy as np
import torch

from tensorrt_llm._torch.mmap_utils import madvise_range
from tensorrt_llm._wide_ep_fault_tolerance import TURN_FAULT_TOLERANCE_OFF
from tensorrt_llm.bindings.internal import runtime
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from .moe_load_balancer import HostMoeTensorSharer, MoeLoadBalancer, SingleLayerMoeLoadBalancer

# The empty spare slots of this many MoE layers are released per call. Each call that releases any
# first waits for the device to finish its queued work, so a call per layer would finish later.
_RELEASE_LAYERS_PER_CALL = 8


def plan_layer_placement(
    current: np.ndarray,
    active: np.ndarray,
    active_slot_count: int,
    expert_count: int,
    has_host_weights: np.ndarray,
    layer_idx: int,
) -> np.ndarray:
    """Return a deterministic ``[ranks, slots]`` placement of every expert on the ``active`` ranks.

    Active ranks keep the experts of their active slots, and while a rank is inactive, the experts
    of their spare slots that no active slot holds. An active slot that repeats an expert of an
    earlier active slot holds an extra replica, which an interrupted online EPLB update or the
    initial placement of a rejoining rank can leave behind. A lost expert takes an empty active slot
    first. Next it takes an extra replica's slot if routing uses one replica per expert or every
    rank is active, else an empty spare slot, and an extra replica's slot only when no spare slot
    is left. With every rank active, the active slots can hold every expert, so every spare slot
    empties. For a spare slot, ranks whose node has no host copy of the expert come first, so that
    a later node loss still leaves a copy on another node.

    Args:
        current: ``[ranks, slots]`` experts, -1 for an empty slot. A rejoining rank's row holds
            the initial placement it loaded.
        active: ``[ranks]`` flags telling whether a rank is active.
        active_slot_count: The active slots per rank, the slots before the spare slots.
        expert_count: The number of experts of the layer.
        has_host_weights: ``[ranks, experts]`` flags telling whether a rank's node keeps host
            weights for an expert.
        layer_idx: The layer index, which rotates the rank order so restored experts spread out.

    Raises:
        ValueError: The active ranks' spare slots and extra replicas cannot hold every expert.
    """
    placement = np.full_like(current, -1)
    placement[active, :active_slot_count] = current[active, :active_slot_count]
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
    # With every rank active, the empty and extra replicas' active slots outnumber the lost
    # experts, so every spare slot empties.
    all_active = bool(active.all())
    replicas = np.bincount(placement[placement >= 0], minlength=expert_count)
    spare_slots = range(active_slot_count, current.shape[1])
    if not all_active:
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
        # Prefer a rank whose node has no host weights for the expert, then the fewest filled spare
        # slots, then the rank order, then a slot that held an expert in ``current`` (its memory
        # may still be backed), then the lowest slot.
        candidates = [
            (
                has_host_weights[rank, expert_id],
                np.count_nonzero(placement[rank, active_slot_count:] >= 0),
                order,
                current[rank, slot] < 0,
                slot,
            )
            for order, rank in enumerate(rank_order)
            for slot in spare_slots
            if placement[rank, slot] < 0
        ]
        if extra_slots and (single_replica or all_active or not candidates):
            placement[extra_slots.popleft()] = expert_id
        elif candidates:
            _, _, order, _, slot = min(candidates)
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

    ``staged`` has one ``(layer, target rank, slot, expert, source rank)`` row per slot that loads
    an expert whose host weights only another node keeps. Every field except ``device_bytes`` is the
    same on every active rank.
    """

    active_ranks: tuple[bool, ...]
    placements: tuple[np.ndarray, ...]
    placement_before: tuple[np.ndarray, ...]
    staged: np.ndarray
    digest: str
    device_bytes: int


class ExpertRestorePlanMismatchError(ValueError):
    """The expert restore plan of a rank differs from rank 0's."""


def _next_piece(half: tuple[int, int], cursor: int, size: int) -> Optional[tuple[int, int]]:
    """Return the next aligned piece of up to ``size`` bytes in ``half``, or None if it is full."""
    begin, end = half
    start = HostMoeTensorSharer.align_size(max(begin, cursor))
    return (start, min(size, end - start)) if start < end else None


class StagingPiece(NamedTuple):
    """Bytes of one weight, indexed by sorted name, that a chunk copies through a workspace row."""

    chunk: int
    layer: int
    target: int
    slot: int
    expert: int
    weight: int
    source: int
    row_offset: int
    weight_offset: int
    size: int

    @property
    def row_range(self) -> slice:
        """Return the workspace row bytes that the piece takes."""
        return slice(self.row_offset, self.row_offset + self.size)

    @property
    def weight_range(self) -> slice:
        """Return the weight bytes that the piece copies."""
        return slice(self.weight_offset, self.weight_offset + self.size)


def staging_pieces(
    staged: np.ndarray, weight_bytes: Sequence[Sequence[int]], row_bytes: int
) -> list[StagingPiece]:
    """Return the pieces that copy the ``staged`` slots through workspace rows of ``row_bytes``.

    Per target, chunks alternate between the two halves of the row, so that a chunk's writes miss
    the previous chunk's reads.

    Args:
        staged: The staged slots, as in ``ExpertRestorePlan.staged``.
        weight_bytes: Per layer, a slot's weight sizes in bytes, by sorted weight name.

    Raises:
        ValueError: A workspace row leaves no aligned room to stage in.
    """
    split = row_bytes // 2
    halves = ((0, split), (split, row_bytes))
    if not all(end > HostMoeTensorSharer.align_size(begin) for begin, end in halves):
        raise ValueError(f"A workspace row of {row_bytes} bytes leaves no aligned room to stage in")
    chunks: dict[int, int] = collections.defaultdict(int)
    cursors: dict[int, int] = collections.defaultdict(int)
    pieces = []
    for layer, target, slot, expert, source in staged.tolist():
        for weight, size in enumerate(weight_bytes[layer]):
            done = 0
            while done < size:
                piece = _next_piece(halves[chunks[target] % 2], cursors[target], size - done)
                if piece is None:
                    chunks[target] += 1
                    cursors[target] = 0
                    continue
                offset, length = piece
                weight_piece = (weight, source, offset, done, length)
                pieces.append(
                    StagingPiece(chunks[target], layer, target, slot, expert, *weight_piece)
                )
                cursors[target] = offset + length
                done += length
    return sorted(pieces, key=lambda piece: piece.chunk)


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
    ``register_weight_slots_after_to_cuda`` checks this for the weights whose spare slots map no
    memory.
    """
    return [
        (name, param)
        for name, param in module.named_parameters(recurse=False)
        if param.dim() > 0 and param.shape[0] == module.expert_size_per_partition
    ]


def create_spare_slot_weights(module: torch.nn.Module) -> None:
    """Move the slot weights of ``module`` to unbacked spare slots.

    The node's host weight experts are chosen before the slot weights move. A replacement that
    reattaches its rank's host weights for this layer loads none.
    """
    layer = module.layer_load_balancer
    weights = slot_parameters(module)
    sharer = layer.host_tensor_sharer
    sharer.keep_next_node_experts(
        layer.initial_placement.reshape(-1, layer.total_slot_count_per_rank)
    )
    sharer.reattach_own_shm({name: (param.dtype, param.shape[1:]) for name, param in weights})
    for name, param in weights:
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


def _keep_after_exit(shm: shared_memory.SharedMemory) -> None:
    """Keep ``shm`` after this process exits, until the node supervisor unlinks it."""
    # The resource tracker registers POSIX shared memory under its name with the leading slash.
    resource_tracker.unregister(f"/{shm.name}", "shared_memory")


def next_node_experts(placement: np.ndarray, ranks: Sequence[int]) -> np.ndarray:
    """Return the sorted experts that the ``[ranks, slots]`` ``placement`` puts on ``ranks``."""
    experts = placement[list(ranks)]
    return np.unique(experts[experts >= 0])


_FINGERPRINT_BYTES = 32
SHARED_MEMORY_DIR = "/dev/shm"  # nosec B108 - the POSIX shared-memory directory, not a temporary file


class DeploymentHostWeights(HostMoeTensorSharer):
    """Host tensor sharer whose shared memory a replacement reattaches after its process exits.

    With ``next_node_ranks``, the node keeps host weights only for the experts on those ranks, so
    each expert has host weights on the node before its own. Otherwise it keeps all of them.
    """

    data_offset = HostMoeTensorSharer.align_size(_FINGERPRINT_BYTES)

    def __init__(
        self, *args, replacement: bool, next_node_ranks: Optional[Sequence[int]] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.replacement = replacement
        self.next_node_ranks = next_node_ranks
        self.reattached_own_shm = False
        # The host weights on this node by expert and weight name, which restores load slots from.
        self.node_host_weights: dict[tuple[int, str], torch.Tensor] = {}

    def keep_next_node_experts(self, placement: np.ndarray) -> None:
        """Keep host weights for the experts that ``placement`` puts on the next node."""
        if self.next_node_ranks is not None:
            self.keep_experts(next_node_experts(placement, self.next_node_ranks).tolist())

    def reattach_own_shm(self, name_info: dict[str, tuple[torch.dtype, torch.Size]]) -> bool:
        """Return whether this replacement reattaches its failed rank's shared memory.

        It reattaches the shared memory if it holds host weights matching ``name_info``. It then
        loads none and keeps the shared memory mapped, whatever later happens to its name.
        Otherwise it writes its host weights to new shared memory.
        """
        if self.replacement:
            shm = self._attach(self.local_rank, name_info)
            if shm is not None:
                self.own_shm = shm
                self.reattached_own_shm = True
                self.own_expert_ids = []
        return self.reattached_own_shm

    def _create_own_shm(self, size: int) -> shared_memory.SharedMemory:
        if self.reattached_own_shm:
            return self.own_shm
        shm = super()._create_own_shm(size)
        _keep_after_exit(shm)
        return shm

    def finalize_layer_weights(self) -> None:
        super().finalize_layer_weights()
        if not self.reattached_own_shm:
            # Written last, so a process that finds the fingerprint also finds the host weights.
            self.own_shm.buf[:_FINGERPRINT_BYTES] = self._fingerprint(
                self.local_rank, self.name_info
            )

    def finalize_host_tensor_sharing(self, add_host_weight_fn: Optional[Callable] = None) -> None:
        if self.reattached_own_shm:
            self._import_host_weights(self.own_shm, self.local_rank)

        def keep(expert_id: int, name: str, weight: torch.Tensor) -> None:
            self.node_host_weights[(expert_id, name)] = weight
            if add_host_weight_fn is not None:
                add_host_weight_fn(expert_id, name, weight)

        super().finalize_host_tensor_sharing(keep)
        # Unmap the host weights this rank wrote. They stay in /dev/shm and fault back in on read.
        madvise_range(
            ctypes.addressof(ctypes.c_char.from_buffer(self.own_shm.buf)), self.own_shm.size
        )

    def _open_peer_shm(self, rank: int) -> shared_memory.SharedMemory:
        shm = self._attach(rank, self.name_info)
        if shm is None:
            raise RuntimeError(
                f"EPLB shared memory {self.get_shared_memory_name(rank)} is missing or does not "
                "hold its host weights"
            )
        return shm

    def pre_shutdown_cleanup(self) -> None:
        # Drop the tensor views of the shared memory before closing it, so that no view points
        # to unmapped memory.
        self.host_weights.clear()
        self.node_host_weights.clear()
        for shm in self.imported_shms:
            shm.close()
        self.imported_shms = []
        super().pre_shutdown_cleanup()

    def post_shutdown_cleanup(self):
        self.own_shm = None
        super().post_shutdown_cleanup()

    def node_bytes(self, slot_bytes: list[int]) -> int:
        """Return the bytes that all local ranks' shared memory for this layer takes in /dev/shm."""
        return (
            len(self.expert_ids) * sum(self.align_size(size) for size in slot_bytes)
            + self.local_size * self.data_offset
        )

    def node_bytes_present(self) -> int:
        """Return the bytes this layer's existing shared memory on the node takes in /dev/shm."""
        present = 0
        for rank in range(self.local_size):
            try:
                path = os.path.join(SHARED_MEMORY_DIR, self.get_shared_memory_name(rank))
                present += os.stat(path).st_blocks * 512
            except FileNotFoundError:
                pass
        return present

    def _fingerprint(
        self, rank: int, name_info: dict[str, tuple[torch.dtype, torch.Size]]
    ) -> bytes:
        layout = (
            self.layer_id,
            self.expert_count,
            self.expert_ids,
            self.local_size,
            rank,
            [
                (name, str(dtype), tuple(shape))
                for name, (dtype, shape) in sorted(name_info.items())
            ],
        )
        return hashlib.sha256(repr(layout).encode()).digest()

    def _attach(
        self, rank: int, name_info: dict[str, tuple[torch.dtype, torch.Size]]
    ) -> Optional[shared_memory.SharedMemory]:
        """Return the intact shared memory of local ``rank`` if this user owns it with mode 0600.

        Shared memory is intact once its fingerprint, written last, matches ``name_info``. Shared
        memory that its rank has not sized yet, or that another rank removes meanwhile, is not.
        """
        try:
            # Shared memory of size 0 cannot be mapped (ValueError).
            shm = shared_memory.SharedMemory(name=self.get_shared_memory_name(rank))
        except (FileNotFoundError, ValueError):
            return None
        _keep_after_exit(shm)
        # Check the opened descriptor, since the path may name another file by now.
        status = os.fstat(shm._fd)
        if (
            status.st_nlink > 0
            and status.st_uid == os.getuid()
            and stat.S_IMODE(status.st_mode) == 0o600
            and bytes(shm.buf[:_FINGERPRINT_BYTES]) == self._fingerprint(rank, name_info)
        ):
            return shm
        shm.close()
        return None

    def holds_host_weights(self, rank: int) -> bool:
        """Return whether the shared memory of local ``rank`` holds that rank's host weights."""
        shm = self._attach(rank, self.name_info)
        if shm is None:
            return False
        shm.close()
        return True


class EplbExpertRestore:
    """Size the spare slots of a ``MoeLoadBalancer`` and restore failed ranks' experts into them."""

    def __init__(self, balancer: "MoeLoadBalancer") -> None:
        self.balancer = balancer
        nodes = self._rank_nodes()
        self.largest_failure = max(map(len, self._recoverable_failures(nodes)), default=0)
        order = sorted(set(nodes))
        next_node = {node: order[(i + 1) % len(order)] for i, node in enumerate(order)}
        # Per rank, the next node's ranks in node order. The first node follows the last one.
        self._next_node_ranks = [
            [rank for rank, node in enumerate(nodes) if node == next_node[rank_node]]
            for rank_node in nodes
        ]
        if (
            self.largest_failure
            and balancer.layer_updates_per_iter > 0
            and not runtime.host_accessible_device_memory_uses_gdrcopy()
        ):
            raise ValueError(
                "Wide-EP fault tolerance with online EPLB requires GDRCopy, because online EPLB's "
                "CPU copies reach the spare slots only through GDRCopy. EPLB uses GDRCopy if it is "
                "installed and either GPU memory is not a NUMA node (as in CDMM mode) or "
                "TRTLLM_EPLB_FORCE_GDRCOPY=1 is set. Otherwise set layer_updates_per_iter=0 or "
                f"{TURN_FAULT_TOLERANCE_OFF}."
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

    def next_node_ranks(self) -> Optional[list[int]]:
        """Return the next node's ranks, or None with online EPLB, where nodes keep all experts."""
        if self.balancer.layer_updates_per_iter > 0:
            return None
        return self._next_node_ranks[self.balancer.ep_rank]

    def _host_weight_ranks(self, layer: SingleLayerMoeLoadBalancer) -> np.ndarray:
        """Return ``[ranks, experts]`` flags telling whether a rank's node keeps host weights."""
        ep_size = self.balancer.ep_size
        has_host_weights = np.zeros((ep_size, layer.expert_count), dtype=bool)
        if not layer.need_load_shared_weights():
            return has_host_weights
        if layer.updates_enabled:
            has_host_weights[:] = True
            return has_host_weights
        placement = layer.initial_placement.reshape(ep_size, -1)
        for rank in range(ep_size):
            has_host_weights[rank, next_node_experts(placement, self._next_node_ranks[rank])] = True
        return has_host_weights

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

    def largest_failure_restore_bytes(self) -> int:
        """Return the most device memory this rank backs to restore any recoverable failure."""
        ep_rank, ep_size = self.balancer.ep_rank, self.balancer.ep_size
        current = [np.array(ranks, dtype=np.int32) for _, ranks in self.current_placement()]
        largest = 0
        for failure in self._recoverable_failures(self._rank_nodes()):
            if ep_rank in failure:
                continue
            active_ranks = [rank not in failure for rank in range(ep_size)]
            largest = max(largest, self.plan_expert_restore(active_ranks, current).device_bytes)
        return largest

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

    def _host_tensor_sharers(self) -> list[DeploymentHostWeights]:
        return [
            layer.host_tensor_sharer
            for layer in self.balancer.single_layer_load_balancers
            if layer.need_load_shared_weights()
        ]

    def node_ranks_without_host_weights(self) -> list[int]:
        """Return the other ranks of this node whose shared memory lacks their host weights."""
        ep_rank = self.balancer.ep_rank
        sharers = self._host_tensor_sharers()
        nodes = self._rank_nodes()
        node_ranks = [rank for rank, node in enumerate(nodes) if node == nodes[ep_rank]]
        return [
            rank
            for local_rank, rank in enumerate(node_ranks)
            if rank != ep_rank
            and not all(sharer.holds_host_weights(local_rank) for sharer in sharers)
        ]

    def plan_expert_restore(
        self,
        active_ranks: Sequence[bool],
        current: Sequence[np.ndarray],
    ) -> ExpertRestorePlan:
        """Plan every layer's placement for ``active_ranks``.

        With static EPLB and every rank active, the active slots take the initial placement.

        Args:
            active_ranks: Whether each EP rank is active.
            current: The per-layer placement that the active ranks agreed on, so that every
                active rank plans the same placement. An active rank's row is its row of
                ``current_placement``, so a slot that a stopped restore wrote loads again, and a
                rejoining rank's row is the initial placement it loaded.

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
        digest = hashlib.sha256(active.tobytes())
        placements = []
        placements_before = []
        staged = []
        # Each set of source ranks takes turns, so that the copies spread over them.
        turns: dict[tuple[int, ...], int] = collections.defaultdict(int)
        device_bytes = 0
        for i, (layer, layer_current) in enumerate(
            zip(balancer.single_layer_load_balancers, current, strict=True)
        ):
            before = np.asarray(layer_current, dtype=np.int32)
            planned = before
            if active.all() and not layer.updates_enabled:
                # Static EPLB keeps the initial placement, so a full rejoin restores it, also in
                # the extra replicas' slots that lost experts took.
                active_slots = layer.active_slot_count_per_rank
                planned = before.copy()
                planned[:, :active_slots] = layer.initial_placement.reshape(before.shape)[
                    :, :active_slots
                ]
            has_host_weights = self._host_weight_ranks(layer)
            placement = plan_layer_placement(
                planned,
                active,
                layer.active_slot_count_per_rank,
                layer.expert_count,
                has_host_weights,
                i,
            )
            for target, slot in zip(*np.nonzero((placement >= 0) & (placement != before))):
                expert = placement[target, slot]
                if has_host_weights[target, expert]:
                    continue
                sources = tuple(np.flatnonzero(has_host_weights[:, expert] & active).tolist())
                if not sources:
                    holders = np.flatnonzero(has_host_weights[:, expert]).tolist()
                    raise ValueError(
                        f"No active rank keeps host weights for expert {expert} of MoE layer "
                        f"{layer.get_layer_idx()}: only ranks {holders} keep them, and none of them "
                        "is active. With static EPLB, each node keeps the host weights of the next "
                        "node's experts, so after the loss of a node, the next node's experts have "
                        "no host copy until a rank of the lost node rejoins"
                    )
                staged.append((i, target, slot, expert, sources[turns[sources] % len(sources)]))
                turns[sources] += 1
            device_bytes += spare_slot_bytes_to_back(
                layer, filled_spare_slots(layer, placement[balancer.ep_rank])
            )
            digest.update(placement.tobytes())
            placements.append(placement)
            placements_before.append(before)
        staged = np.array(staged, dtype=np.int64).reshape(-1, 5)
        digest.update(staged.tobytes())
        return ExpertRestorePlan(
            tuple(bool(rank) for rank in active),
            tuple(placements),
            tuple(placements_before),
            staged,
            digest.hexdigest()[:16],
            device_bytes,
        )

    def set_spare_slots_backed(self, layer_slots: Sequence[Sequence[int]], backed: bool) -> int:
        """Back spare slots with device memory or map them back to the shared scratch.

        Returns:
            The number of bytes whose backing changed.
        """
        changed = self._set_backed(layer_slots, backed)
        self.backed_spare_slot_bytes += changed if backed else -changed
        return changed

    def empty_spare_slots(self) -> list[list[int]]:
        """Return, by layer, the spare slots that the installed placement leaves empty.

        An active slot that a stopped restore emptied is unusable too, but it has no unbacked
        memory to release.

        Raises:
            RuntimeError: An iteration is active.
        """
        balancer = self.balancer
        if balancer.in_iter:
            raise RuntimeError("Cannot read the empty spare slots while an iteration is active")
        return [
            [slot for slot in layer.get_unusable_local_slots() if slot in layer.spare_slots]
            for layer in balancer.single_layer_load_balancers
        ]

    def release_spare_slots(self, layer_slots: Sequence[Sequence[int]]) -> int:
        """Map the empty spare slots ``layer_slots``, by layer, back to the shared scratch.

        It may run on a worker thread while forwards and online EPLB updates run: no token reaches
        an empty spare slot, no kernel reads the weights of a slot without tokens, no update fills
        these slots, and no CUDA graph is captured outside warmup (``allow_capture`` in
        ``cuda_graph_runner.py``). It leaves ``backed_spare_slot_bytes`` to the caller's thread.

        Returns:
            The number of bytes released.
        """
        released = 0
        for start in range(0, len(layer_slots), _RELEASE_LAYERS_PER_CALL):
            end = start + _RELEASE_LAYERS_PER_CALL
            released += self._set_backed(
                [slots if start <= i < end else [] for i, slots in enumerate(layer_slots)], False
            )
        return released

    def _set_backed(self, layer_slots: Sequence[Sequence[int]], backed: bool) -> int:
        """Back or unback ``layer_slots`` in one call and return the bytes whose backing changed."""
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
        return torch.ops.trtllm.moe_unbacked_slots_set_backed(tensors, slots, slot_counts, backed)

    def restore_lost_experts(
        self,
        plan: ExpertRestorePlan,
        workspace: torch.Tensor,
        allgather: Callable[[Any], Sequence[Any]],
        times: Optional[dict[str, float]] = None,
        progress: Callable[[], None] = lambda: None,
    ) -> int:
        """Load the slots that ``plan`` changes on this rank, install it and return the bytes mapped.

        Every active rank must restore the same plan before any rank routes with it.

        Args:
            plan: The plan that every active rank restores.
            workspace: The ``[ep_size, bytes]`` uint8 all-to-all workspace, which no rank may use
                and no former member may write until the restore returns.
            allgather: Return the values that every active rank passes to it, by rank, with None
                for an inactive rank.
            times: Receives each phase's duration in milliseconds.
            progress: Called after each staged chunk and each layer loaded locally, so that a long
                restore keeps reporting progress. It may raise to stop the restore. The slots that
                ``plan`` loads on this rank then hold no expert in ``current_placement`` until a
                restore installs a placement.

        Raises:
            ExpertRestorePlanMismatchError: The plan or the workspace row size of this rank differs
                from the lowest active rank's.
            RuntimeError: An iteration is active, or the online EPLB worker may still write the
                slots.
            ValueError: A workspace row leaves no aligned room to stage in.
        """
        balancer = self.balancer
        if balancer.in_iter:
            raise RuntimeError("Cannot restore experts while an iteration is active")
        # The lowest active rank is rank 0, which never rejoins. A rank with another plan or other
        # workspace rows, which split the staging into other chunks, stops before it writes a slot
        # or a workspace row.
        layout = (plan.digest, workspace.shape[1])
        agreed = tuple(next(value for value in allgather(layout) if value is not None))
        if agreed != layout:
            raise ExpertRestorePlanMismatchError(
                f"EP rank {balancer.ep_rank} planned expert placement {plan.digest} with workspace "
                f"rows of {layout[1]} bytes, but the lowest active rank planned {agreed[0]} with "
                f"rows of {agreed[1]} bytes"
            )
        # No EPLB weight update may write the slots that the restore loads, or swap the placement
        # that the check below reads.
        balancer.load_balancer_impl.check_slots_writable()
        layers = balancer.single_layer_load_balancers
        # Staging is collective, so no rank takes this fast path while the plan stages a slot.
        if (
            not plan.staged.size
            and not self._invalid_slots
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
        # One call backs every layer: each call waits for all queued device work, so per-layer calls
        # would also wait for the loads of the layers before them.
        layer_slots = [
            filled_spare_slots(layer, placement[balancer.ep_rank])
            for layer, placement in zip(layers, plan.placements)
        ]
        op_start_s = time.perf_counter()
        mapped = self.set_spare_slots_backed(layer_slots, True)
        op_done_s = time.perf_counter()
        start_s = _phase_done(times, "back", start_s)
        if times is not None:
            times["back op"] = (op_done_s - op_start_s) * 1e3
        loaded_layers = 0

        def load_local_slots(end: int) -> None:
            """Load the local slots of the layers before ``end`` that are not loaded yet."""
            nonlocal loaded_layers
            for i in range(loaded_layers, end):
                self._load_local_slots(plan, i, progress)
            loaded_layers = max(loaded_layers, end)

        self._stage_slots(plan.staged, workspace, allgather, load_local_slots, progress)
        load_local_slots(len(layers))
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

    def _load_local_slots(
        self, plan: ExpertRestorePlan, layer_idx: int, progress: Callable[[], None]
    ) -> None:
        """Load the unstaged slots that ``plan`` changes on this rank in layer ``layer_idx``.

        They are copied from this node's host weights.
        """
        ep_rank = self.balancer.ep_rank
        layer = self.balancer.single_layer_load_balancers[layer_idx]
        staged = plan.staged[(plan.staged[:, 0] == layer_idx) & (plan.staged[:, 1] == ep_rank), 2]
        changed = self._loaded_slots(plan, layer_idx)
        slots = np.flatnonzero(changed & ~np.isin(np.arange(changed.size), staged))
        for slot in slots:
            expert = int(plan.placements[layer_idx][ep_rank, slot])
            for name, tensor in layer.slot_tensors.items():
                host = layer.host_tensor_sharer.node_host_weights[(expert, name)]
                _bytes(tensor[slot]).copy_(_bytes(host))
        progress()

    def _stage_slots(
        self,
        staged: np.ndarray,
        workspace: torch.Tensor,
        allgather: Callable[[Any], Sequence[Any]],
        load_local_slots: Callable[[int], None],
        progress: Callable[[], None],
    ) -> None:
        """Copy the ``staged`` experts into their slots through ``workspace``, chunk by chunk.

        Every active rank meets in ``allgather`` between the sources' writes and the targets' reads.

        Args:
            load_local_slots: Loads the local slots of the layers before the given one. Local loads
                need no peer, so before each chunk's allgather this rank loads those up to the
                chunk's last layer, while the other ranks may still be backing their slots.
        """
        if not staged.size:
            return
        start_s = time.perf_counter()
        ep_rank = self.balancer.ep_rank
        layers = self.balancer.single_layer_load_balancers
        names = [sorted(layer.slot_tensors) for layer in layers]
        weight_bytes = [
            [layer.slot_tensors[name][0].nbytes for name in layer_names]
            for layer, layer_names in zip(layers, names)
        ]
        pieces = staging_pieces(staged, weight_bytes, workspace.shape[1])
        stream = torch.cuda.current_stream()
        for chunk, chunk_pieces in itertools.groupby(pieces, key=lambda piece: piece.chunk):
            chunk_pieces = list(chunk_pieces)
            for piece in chunk_pieces:
                if piece.source == ep_rank:
                    name = names[piece.layer][piece.weight]
                    host = layers[piece.layer].host_tensor_sharer.node_host_weights[
                        (piece.expert, name)
                    ]
                    workspace[piece.target, piece.row_range].copy_(_bytes(host)[piece.weight_range])
            load_local_slots(max(piece.layer for piece in chunk_pieces) + 1)
            stream.synchronize()
            allgather(None)
            progress()
            for piece in chunk_pieces:
                if piece.target == ep_rank:
                    name = names[piece.layer][piece.weight]
                    slot_weight = layers[piece.layer].slot_tensors[name][piece.slot]
                    _bytes(slot_weight)[piece.weight_range].copy_(
                        workspace[piece.target, piece.row_range]
                    )
        stream.synchronize()
        _, targets, _, _, sources = staged.T
        copied = sum(piece.size for piece in pieces if ep_rank in (piece.source, piece.target))
        logger.info(
            f"Rank {ep_rank} staged {np.count_nonzero(sources == ep_rank)} experts out and "
            f"{np.count_nonzero(targets == ep_rank)} in ({copied} bytes) in "
            f"{pieces[-1].chunk + 1} workspace chunks in "
            f"{(time.perf_counter() - start_s) * 1e3:.0f} ms"
        )


class FaultTolerantMoeLoadBalancer(MoeLoadBalancer):
    """MoE load balancer whose spare slots take the experts of failed ranks."""

    fault_tolerance = True

    def __init__(
        self,
        *args,
        mapping: Mapping,
        replacement: bool,
        wait_for_host_weights: Callable[[int, Callable[[], list[int]]], None],
        **kwargs,
    ) -> None:
        """Create the load balancer of one rank of a deployment.

        Args:
            mapping: The mapping whose nodes decide which failures are recoverable.
            replacement: Whether this rank replaces a failed one and reattaches the host weights on
                its node.
            wait_for_host_weights: On a replacement, wait until the other ranks on its node hold
                their host weights. It takes this rank and a callable that returns the node's
                ranks without them.
        """
        super().__init__(*args, **kwargs)
        self.mapping = mapping
        self.expert_restore = EplbExpertRestore(self)
        self.host_tensor_sharer_cls = functools.partial(
            DeploymentHostWeights,
            replacement=replacement,
            next_node_ranks=self.expert_restore.next_node_ranks(),
        )
        self.replacement = replacement
        self._wait_for_host_weights = wait_for_host_weights

    def _spare_slot_count(self, expert_count: int, slot_count_per_rank: int) -> int:
        return self.expert_restore.spare_slot_count(expert_count, slot_count_per_rank)

    @property
    def adds_spare_slots(self) -> bool:
        # A layer gets spare slots exactly when some failure is recoverable.
        return self.expert_restore.largest_failure > 0

    def finalize_model(self) -> None:
        if self.replacement:
            self._wait_for_host_weights(
                self.ep_rank, self.expert_restore.node_ranks_without_host_weights
            )
        super().finalize_model()
        if self.replacement:
            reattached = sum(
                layer.host_tensor_sharer.reattached_own_shm
                for layer in self.single_layer_load_balancers
            )
            logger.info(
                f"Rank {self.ep_rank} reattached the host weights it wrote before it failed for "
                f"{reattached} of {len(self.single_layer_load_balancers)} MoE layers"
            )
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
