# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The R4 rank hierarchy shared by the CUDA scheduler and the SAMI copy."""

from __future__ import annotations

__all__ = ["MAX_EP", "MIN_EP", "MIN_GROUP_SIZE", "hierarchy_group_sizes"]

MIN_EP = 2
MAX_EP = 32
MIN_GROUP_SIZE = 4


def hierarchy_group_sizes(world: int) -> tuple[int, ...]:
    """Return HALO-M's R4 aligned hierarchy, coarsest to finest.

    Level 0 is the whole EP world.  Later levels halve the group size only
    while the world divides exactly and every group still holds at least
    ``MIN_GROUP_SIZE`` ranks.  ``halo_q_scheduler.cu::hierarchy_depth`` is the
    device-side statement of the same ladder; the two must agree for every
    world this package accepts, so there is deliberately one host definition
    rather than one per subpackage.
    """

    if (type(world) is not int or world < MIN_EP or world > MAX_EP
            or world % 2):
        raise ValueError(
            f"hierarchical SAMI supports even EP in [{MIN_EP},{MAX_EP}]")
    result = [world]
    groups = 2
    while world % groups == 0 and world // groups >= MIN_GROUP_SIZE:
        result.append(world // groups)
        groups *= 2
    return tuple(result)
