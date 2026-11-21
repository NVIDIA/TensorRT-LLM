# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose safe context boundaries for periodic in-forward snapshots."""

from typing import TYPE_CHECKING, NamedTuple, Sequence

if TYPE_CHECKING:
    import torch


class SnapshotChunkPlan(NamedTuple):
    end: int
    points: tuple[int, ...]


class PeriodicSnapshotReservation(NamedTuple):
    start: int
    end: int
    interval: int
    points: tuple[int, ...]
    slots: tuple[int, ...]


class PeriodicSnapshotBatch(NamedTuple):
    """Device indices and buffer size for one context step."""

    interval: int
    checkpoint_count: int
    checkpoint_cu_starts: "torch.Tensor"
    checkpoint_rows: "torch.Tensor"
    positions: "torch.Tensor"
    conv_indices: "torch.Tensor"
    slots: "torch.Tensor"


class SnapshotStepIndices(NamedTuple):
    checkpoint_count: int
    checkpoint_rows: tuple[int, ...]
    positions: tuple[int, ...]
    conv_indices: tuple[int, ...]
    slots: tuple[int, ...]


def snapshot_step_indices(
    reservation: PeriodicSnapshotReservation, conv_width: int
) -> SnapshotStepIndices:
    """Build checkpoint and convolution indices for a single packed context."""
    start, end, interval, points, slots = reservation
    positions = tuple(point - start for point in points)
    if (
        start < 0
        or end <= start
        or interval <= 0
        or not points
        or len(points) != len(slots)
        or tuple(sorted(set(points))) != points
        or len(set(slots)) != len(slots)
        or any(slot < 0 for slot in slots)
        or any(
            position < conv_width or position >= end - start or position % interval
            for position in positions
        )
    ):
        raise ValueError("Invalid periodic snapshot reservation")
    return SnapshotStepIndices(
        checkpoint_count=(end - start) // interval,
        checkpoint_rows=tuple(position // interval - 1 for position in positions),
        positions=positions,
        conv_indices=tuple(
            index for position in positions for index in range(position - conv_width, position)
        ),
        slots=slots,
    )


def plan_periodic_snapshot_chunk(
    start: int,
    end: int,
    expected_points: Sequence[int],
    interval: int,
    alignment: int,
    max_pending: int,
) -> SnapshotChunkPlan:
    """Plan the interior snapshots that a context chunk can safely cross.

    ``end`` is the desired absolute end after the normal token-budget and
    model constraints. Unsupported points remain chunk boundaries. A point at
    the returned end is saved by the ordinary end-of-chunk path, so it is not
    included in the pending points.
    """
    aligned = interval > 0 and interval % alignment == 0 and start % interval == 0
    if aligned:
        # FlashInfer emits a checkpoint every interval, even when only some
        # of those boundaries are reusable snapshot points.
        end = min(end, start + (max_pending + 1) * interval)
    pending = []
    for point in sorted(set(expected_points)):
        if point <= start:
            continue
        if point >= end:
            break
        if not aligned or point % interval != 0 or len(pending) >= max_pending:
            return SnapshotChunkPlan(point, tuple(pending))
        pending.append(point)
    return SnapshotChunkPlan(end, tuple(pending))
