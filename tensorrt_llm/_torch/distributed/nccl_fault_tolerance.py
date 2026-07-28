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

import operator
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
_reconfiguration_lineages: dict[_Group, _Group] = {}
_completed_reconfigurations: dict[_Group, tuple[int, _Group]] = {}
_latest_reconfiguration_attempts: dict[_Group, tuple[int | None, _Group]] = {}
_unavailable_reconfiguration_lineages: dict[_Group, tuple[int | None, _Group]] = {}


def _canonical_integer(value: object, name: str) -> int:
    """Return an exact integral value without accepting bools or coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _canonical_group(group: Iterable[int], name: str) -> _Group:
    ranks = tuple(_canonical_integer(rank, f"{name} entries") for rank in group)
    if not ranks:
        raise ValueError(f"{name} must not be empty")
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"{name} must not contain duplicate ranks")
    return ranks


def _canonical_recovery_generation(generation: int | None) -> int | None:
    """Validate a coordinator generation accepted by the Torch ``int`` ABI."""
    if generation is None:
        return None
    try:
        canonical_generation = _canonical_integer(generation, "generation")
    except ValueError as error:
        raise ValueError(
            f"generation must be a nonnegative integer no greater than {_MAX_RECOVERY_GENERATION}"
        ) from error
    if canonical_generation < 0 or canonical_generation > _MAX_RECOVERY_GENERATION:
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
    root = _reconfiguration_lineages.get(ranks, ranks)
    unavailable = _unavailable_reconfiguration_lineages.get(root)
    if unavailable is not None:
        generation, requested = unavailable
        generation_description = (
            "legacy generation" if generation is None else f"generation {generation}"
        )
        raise RuntimeError(
            "NCCL error: communicator recovery "
            f"{generation_description} to {list(requested)} did not complete; "
            "advance the coordinator generation and successfully rebuild the "
            "communicator before resuming communication"
        )
    return list(_active_groups.get(root, root))


def is_nccl_group_reconfigured(group: Iterable[int]) -> bool:
    """Cheap hot-path membership-change check for a trusted rank group."""
    if not NCCL_FAULT_TOLERANCE_ENABLED:
        return False
    original_group = group if isinstance(group, tuple) else tuple(group)
    root = _reconfiguration_lineages.get(original_group, original_group)
    if root in _unavailable_reconfiguration_lineages:
        return True
    active_group = _active_groups.get(root)
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
    root = _reconfiguration_lineages.get(original_group, original_group)
    unavailable = _unavailable_reconfiguration_lineages.get(root)
    if unavailable is not None:
        generation, requested = unavailable
        generation_description = (
            "legacy generation" if generation is None else f"generation {generation}"
        )
        raise RuntimeError(
            "NCCL error: communicator recovery "
            f"{generation_description} to {list(requested)} did not complete; "
            f"{operation} cannot resume until the coordinator advances the "
            "generation and the communicator rebuild succeeds"
        )
    active_group = _active_groups.get(root)
    if active_group is not None and active_group != original_group:
        raise RuntimeError(
            "NCCL error: "
            f"{operation} cannot use survivor-only communicator {list(active_group)} "
            f"for statically sharded group {list(original_group)}; redistribute the "
            "missing rank's state and rebuild the mapping before resuming"
        )


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
        root = _reconfiguration_lineages.get(original, original)
        current = _active_groups.get(root, root)
        completed = _completed_reconfigurations.get(root)
        unavailable = _unavailable_reconfiguration_lineages.get(root)
        if generation is not None and completed is not None and unavailable is None:
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

        latest_attempt = _latest_reconfiguration_attempts.get(root)
        if latest_attempt is not None:
            attempted_generation, attempted_target = latest_attempt
            if generation is None:
                raise RuntimeError(
                    "NCCL error: generation is required after a previous "
                    "communicator recovery attempt; advance the coordinator "
                    "generation before retrying"
                )
            if attempted_generation is not None and generation <= attempted_generation:
                if generation == attempted_generation and requested != attempted_target:
                    raise RuntimeError(
                        "NCCL error: conflicting communicator recovery target "
                        f"for generation {generation}: attempted "
                        f"{list(attempted_target)}, requested {list(requested)}"
                    )
                raise RuntimeError(
                    "NCCL error: communicator recovery generation "
                    f"{generation} has already been attempted or is stale; "
                    f"use a generation greater than {attempted_generation}"
                )

        # Link the requested survivor tuple to the original communicator before
        # native code can partially consume rendezvous traffic. A new layer
        # constructed from that tuple will still resolve the last successfully
        # published native group after an asymmetric failure.
        aliases = {
            key for key, lineage_root in _reconfiguration_lineages.items() if lineage_root == root
        }
        aliases.update(key for key, value in _active_groups.items() if value == current)
        aliases.update((root, original, current, requested))
        for alias in aliases:
            _reconfiguration_lineages[alias] = root

        # Reserve the wire namespace before native code can partially consume
        # rendezvous traffic. Retain it on failure so only a newer generation
        # can retry on every survivor in this communicator lineage.
        _latest_reconfiguration_attempts[root] = (generation, requested)
        _unavailable_reconfiguration_lineages[root] = (generation, requested)
        rebuild(list(current), list(requested), rendezvous_id)

        for alias in aliases:
            _active_groups[alias] = requested
        if generation is not None:
            _completed_reconfigurations[root] = (generation, requested)
        # Publish active membership before unblocking hot paths. A reader may
        # fail closed briefly while publication is in progress, but can never
        # dispatch through the terminal communicator from a failed attempt.
        _unavailable_reconfiguration_lineages.pop(root, None)
        return list(requested)


def _reset_nccl_group_registry_for_tests() -> None:
    """Clear process-local membership.  Tests only; native state is untouched."""
    with _registry_lock:
        _active_groups.clear()
        _reconfiguration_lineages.clear()
        _completed_reconfigurations.clear()
        _latest_reconfiguration_attempts.clear()
        _unavailable_reconfiguration_lineages.clear()
