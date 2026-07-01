# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local survivor membership for raw NCCL communicators.

The C++ raw-NCCL registry keys communicators by their global-rank set.  After
``nccl_comm_abort_and_reinit`` replaces a communicator, explicitly dynamic
callers must use the replacement set, while statically sharded TP/CP callers
must stop until their missing state is redistributed. This module keeps that
membership and fail-fast decision in one place instead of attaching it to the
MoE communication object that initiated recovery.

ProcessGroup-backed collectives intentionally do not use this registry: their
groups need to be reconstructed by the ProcessGroup owner.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Iterable, List, Tuple

_Group = Tuple[int, ...]
_MAX_RECOVERY_GENERATION = (1 << 63) - 3

# Fault tolerance is a startup-only mode. Cache the flag once so the default
# collective hot path pays only a cheap boolean branch and never consults the
# survivor registry.
NCCL_FAULT_TOLERANCE_ENABLED = os.environ.get("TLLM_FAULT_TOLERANCE_MODE") == "1"

_registry_lock = threading.RLock()
_active_groups: dict[_Group, _Group] = {}
_completed_reconfigurations: dict[_Group, tuple[int, _Group]] = {}


def _canonical_group(group: Iterable[int], name: str) -> _Group:
    ranks = tuple(int(rank) for rank in group)
    if not ranks:
        raise ValueError(f"{name} must not be empty")
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"{name} must not contain duplicate ranks")
    return ranks


def _canonical_recovery_generation(generation: int | None) -> int | None:
    """Validate a coordinator generation accepted by the Torch ``int`` ABI."""
    if generation is None:
        return None
    canonical_generation = int(generation)
    if (
        canonical_generation < 0
        or canonical_generation != generation
        or canonical_generation > _MAX_RECOVERY_GENERATION
    ):
        raise ValueError(
            f"generation must be a nonnegative integer no greater than {_MAX_RECOVERY_GENERATION}"
        )
    return canonical_generation


def _recovery_rendezvous_id(generation: int | None) -> int:
    """Map a recovery generation to its wire ID.

    ID 0 is reserved for initial bootstrap and ID 1 for the legacy one-shot
    membership shrink without a coordinator generation. Explicit generations
    start at ID 2, keeping every namespace distinct on the retained MPI control
    communicator.
    """
    return 1 if generation is None else generation + 2


def resolve_nccl_group(group: Iterable[int]) -> List[int]:
    """Return the latest survivor set for a raw NCCL rank group."""
    if not NCCL_FAULT_TOLERANCE_ENABLED:
        return list(group)
    ranks = _canonical_group(group, "group")
    # Keep the hot-path lookup free of synchronization primitives so Dynamo
    # can guard the dictionary value and recompile when recovery publishes a
    # new group. Recovery quiesces collective submission before publishing.
    return list(_active_groups.get(ranks, ranks))


def resolve_nccl_group_and_rank(group: Iterable[int], world_rank: int) -> tuple[List[int], int]:
    """Resolve a raw NCCL group and the caller's compact communicator rank."""
    active_group = resolve_nccl_group(group)
    try:
        compact_rank = active_group.index(int(world_rank))
    except ValueError as error:
        raise RuntimeError(
            f"NCCL error: world rank {world_rank} is not in active communicator {active_group}"
        ) from error
    return active_group, compact_rank


def is_nccl_group_reconfigured(group: Iterable[int]) -> bool:
    """Cheap hot-path membership-change check for a trusted rank group."""
    if not NCCL_FAULT_TOLERANCE_ENABLED:
        return False
    original_group = group if isinstance(group, tuple) else tuple(group)
    active_group = _active_groups.get(original_group)
    return active_group is not None and active_group != original_group


def assert_nccl_group_not_reconfigured(group: Iterable[int], operation: str) -> None:
    """Reject static rank sharding after transport-only recovery.

    This hot-path check trusts Mapping-owned groups to already be canonical. A
    single dictionary lookup is enough before recovery; full validation stays
    in the control-path APIs that publish membership.
    """
    if not NCCL_FAULT_TOLERANCE_ENABLED:
        return
    original_group = group if isinstance(group, tuple) else tuple(group)
    active_group = _active_groups.get(original_group)
    if active_group is not None and active_group != original_group:
        raise RuntimeError(
            "NCCL error: "
            f"{operation} cannot use survivor-only communicator {list(active_group)} "
            f"for statically sharded group {list(original_group)}; redistribute the "
            "missing rank's state and rebuild the mapping before resuming"
        )


def publish_nccl_group_reconfiguration(
    original_group: Iterable[int],
    current_group: Iterable[int],
    active_group: Iterable[int],
) -> None:
    """Publish a successful native raw-NCCL communicator replacement.

    ``current_group`` makes stale or racing reconfiguration attempts fail
    before they can overwrite newer membership.  Publishing happens only
    after the coordinated native rebuild succeeds.
    """
    original = _canonical_group(original_group, "original_group")
    current = _canonical_group(current_group, "current_group")
    active = _canonical_group(active_group, "active_group")

    if not set(active).issubset(current):
        raise ValueError("active_group must be a subset of current_group")

    with _registry_lock:
        registered = _active_groups.get(original, original)
        if registered == active:
            return
        if registered != current:
            raise RuntimeError(
                "NCCL error: stale communicator membership update: "
                f"expected {list(registered)}, got {list(current)}"
            )

        # Preserve aliases for the model's original mapping and for any
        # survivor-only mapping views already handed to communication helpers.
        aliases = [key for key, value in _active_groups.items() if value == current]
        aliases.extend((original, current, active))
        for alias in aliases:
            _active_groups[alias] = active


def reconfigure_nccl_group(
    original_group: Iterable[int],
    active_group: Iterable[int],
    rebuild: Callable[[List[int], List[int], int], None],
    generation: int | None = None,
) -> List[int]:
    """Serialize a native communicator rebuild and membership publication.

    Recovery is a process-wide safe-point operation.  Holding the registry
    lock across ``rebuild`` prevents two local layers or worker threads from
    rebuilding the same native communicator to different survivor sets before
    either update is published. ``generation`` is the coordinator's monotonic
    recovery-event generation, advanced for every distinct attempt (including
    transport-only retries). It is required to distinguish a same-membership
    transport rebuild from a duplicate callback without consulting rank-local
    watchdog state, which can differ transiently across survivors. A first
    membership shrink may omit it for compatibility, but every retry after a
    native failure must provide a newly advanced generation so the wire
    rendezvous cannot pair different attempts.
    """
    if not NCCL_FAULT_TOLERANCE_ENABLED:
        raise RuntimeError(
            "NCCL error: communicator reinitialization requires TLLM_FAULT_TOLERANCE_MODE=1"
        )

    original = _canonical_group(original_group, "original_group")
    requested = _canonical_group(active_group, "active_group")
    generation = _canonical_recovery_generation(generation)
    rendezvous_id = _recovery_rendezvous_id(generation)

    with _registry_lock:
        current = _active_groups.get(original, original)
        completed = _completed_reconfigurations.get(original)
        if generation is not None and completed is not None:
            completed_generation, completed_target = completed
            if generation < completed_generation:
                return list(current)
            if generation == completed_generation:
                if requested != completed_target:
                    raise RuntimeError(
                        "NCCL error: conflicting communicator recovery target "
                        f"for generation {generation}: completed "
                        f"{list(completed_target)}, requested {list(requested)}"
                    )
                return list(current)

        if not set(requested).issubset(current):
            raise ValueError("active_group must be a subset of the current_group")
        if requested == current:
            if generation is None:
                raise ValueError(
                    "generation is required for same-membership communicator "
                    "recovery so every survivor makes the same rebuild decision"
                )

        rebuild(list(current), list(requested), rendezvous_id)

        aliases = [key for key, value in _active_groups.items() if value == current]
        aliases.extend((original, current, requested))
        for alias in aliases:
            _active_groups[alias] = requested
        if generation is not None:
            _completed_reconfigurations[original] = (generation, requested)
        return list(requested)


def _reset_nccl_group_registry_for_tests() -> None:
    """Clear process-local membership.  Tests only; native state is untouched."""
    with _registry_lock:
        _active_groups.clear()
        _completed_reconfigurations.clear()
