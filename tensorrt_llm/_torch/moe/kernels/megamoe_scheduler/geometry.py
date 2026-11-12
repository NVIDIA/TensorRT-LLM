# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The R4 rank hierarchy shared by the CUDA scheduler and the SAMI copy."""

from __future__ import annotations

__all__ = [
    "MAX_EP",
    "MIN_EP",
    "MIN_GROUP_SIZE",
    "balanced_copy_enabled",
    "hierarchy_group_sizes",
    "pair_team_objects_enabled",
]

MIN_EP = 2
MAX_EP = 32
MIN_GROUP_SIZE = 4
#: Power-of-two worlds up to this size halve down to aligned rank pairs, so
#: HALO-Q's pair records are an ordinary multicast level (the copy ABI holds
#: at most four levels).
MAX_PAIR_LADDER_EP = 16
#: Largest world whose balanced copy reaches rank pairs through team objects
#: ``{owner, 2j, 2j + 1}``.  A GPU binds every multicast object it joins once
#: per helper set and fails near 960 binds: with a helper set per layer, EP8
#: takes 15 binds per layer (915 at 61 layers), EP16 would take 29.
MAX_PAIR_TEAM_EP = 8
#: moeRebalanceTmaGpuPlanBuilder.cuh kBalancedMaxCopies: a layer holds at most
#: world * helper_count / 2 copies, each on two or more ranks.
_BALANCED_MAX_COPIES = 96


def hierarchy_group_sizes(world: int) -> tuple[int, ...]:
    """Return HALO-M's R4 aligned hierarchy, coarsest to finest.

    Level 0 is the whole EP world.  Later levels halve the group size only
    while the world divides exactly and every group still holds at least
    ``MIN_GROUP_SIZE`` ranks, or 2 ranks for a power-of-two world of at most
    ``MAX_PAIR_LADDER_EP``.  ``moeRebalancePlanPolicy.h::hierarchy_depth`` is
    the device-side statement of the same ladder; the two must agree for every
    world this package accepts, so there is deliberately one host definition
    rather than one per subpackage.
    """

    if (type(world) is not int or world < MIN_EP or world > MAX_EP
            or world % 2):
        raise ValueError(
            f"hierarchical SAMI supports even EP in [{MIN_EP},{MAX_EP}]")
    min_size = (2 if world <= MAX_PAIR_LADDER_EP and world & (world - 1) == 0
                else MIN_GROUP_SIZE)
    result = [world]
    groups = 2
    while world % groups == 0 and world // groups >= min_size:
        result.append(world // groups)
        groups *= 2
    return tuple(result)


def balanced_copy_enabled(world: int, helper_count: int) -> bool:
    """Whether the copy splits every HALO-Q copy between its owner and its
    destination members (route bit 8).  It needs a ladder that ends in rank
    pairs, EP >= 4 and room for a layer's copies in the builder; the arena and
    the copy configuration both ask here.
    """
    sizes = hierarchy_group_sizes(world)
    return (
        world >= 4
        and len(sizes) > 1
        and sizes[-1] == 2
        and world * helper_count <= 2 * _BALANCED_MAX_COPIES
    )


def pair_team_objects_enabled(world: int, helper_count: int) -> bool:
    """Whether an owner outside a rank pair writes its share of a balanced copy
    through a team object (one multicast), else with unicast stores into each
    rank of the pair."""
    return balanced_copy_enabled(world, helper_count) and world <= MAX_PAIR_TEAM_EP
