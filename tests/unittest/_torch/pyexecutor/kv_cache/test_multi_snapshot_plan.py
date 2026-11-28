# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.multi_snapshot import (
    PeriodicSnapshotReservation,
    SnapshotChunkPlan,
    plan_periodic_snapshot_chunk,
    snapshot_step_indices,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "start,end,points,interval,alignment,capacity,expected",
    [
        (
            0,
            8192,
            [2048, 4096, 6144, 8192],
            2048,
            64,
            4,
            SnapshotChunkPlan(8192, (2048, 4096, 6144)),
        ),
        (0, 8192, [2048, 4096, 6144, 8192], 2048, 64, 1, SnapshotChunkPlan(4096, (2048,))),
        (0, 8192, [2048, 3000, 4096], 2048, 64, 4, SnapshotChunkPlan(3000, (2048,))),
        (31, 8192, [2048, 4096], 2048, 64, 4, SnapshotChunkPlan(2048, ())),
        (0, 6000, [2000, 4000], 2000, 64, 4, SnapshotChunkPlan(2000, ())),
        (0, 3000, [2048, 4096], 2048, 64, 4, SnapshotChunkPlan(3000, (2048,))),
        (2048, 8192, [2048, 4096, 4096, 6144], 2048, 64, 2, SnapshotChunkPlan(8192, (4096, 6144))),
        (0, 8192, [2048, 4096], 2048, 64, 0, SnapshotChunkPlan(2048, ())),
    ],
)
def test_periodic_snapshot_chunk_plan(start, end, points, interval, alignment, capacity, expected):
    assert (
        plan_periodic_snapshot_chunk(start, end, points, interval, alignment, capacity) == expected
    )


@pytest.mark.cpu_only
def test_snapshot_step_indices_include_the_end_checkpoint_in_buffer_size():
    reservation = PeriodicSnapshotReservation(2048, 10240, 2048, (4096, 6144, 8192), (7, 3, 9))
    result = snapshot_step_indices(reservation, conv_width=3)
    assert result.checkpoint_count == 4
    assert result.checkpoint_rows == (0, 1, 2)
    assert result.positions == (2048, 4096, 6144)
    assert result.conv_indices == (2045, 2046, 2047, 4093, 4094, 4095, 6141, 6142, 6143)
    assert result.slots == (7, 3, 9)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "reservation,width",
    [
        (PeriodicSnapshotReservation(0, 256, 64, (64,), ()), 3),
        (PeriodicSnapshotReservation(0, 256, 64, (65,), (1,)), 3),
        (PeriodicSnapshotReservation(0, 256, 64, (256,), (1,)), 3),
        (PeriodicSnapshotReservation(0, 256, 64, (64,), (1,)), 65),
        (PeriodicSnapshotReservation(0, 256, 64, (64, 128), (1, 1)), 3),
    ],
)
def test_snapshot_step_indices_reject_invalid_reservations(reservation, width):
    with pytest.raises(ValueError, match="Invalid periodic snapshot reservation"):
        snapshot_step_indices(reservation, width)
