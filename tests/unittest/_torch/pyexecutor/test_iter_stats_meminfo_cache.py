# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The iteration stats' device-memory reading is refreshed at most once per period (CPU only)."""

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import IterStatsMemInfo

pytestmark = pytest.mark.cpu_only


class _Fake:
    def __init__(self):
        self.now = 0
        self.reads = 0

    def clock(self):
        return self.now

    def read(self):
        self.reads += 1
        return (100 - self.reads, 1000)


def test_reuses_reading_within_period():
    f = _Fake()
    m = IterStatsMemInfo(1.0, read=f.read, clock=f.clock)
    assert m(0) == (99, 1000)
    f.now = 999_999_999
    assert m(0) == (99, 1000) and f.reads == 1
    f.now = 1_000_000_000
    assert m(0) == (98, 1000) and f.reads == 2


@pytest.mark.parametrize("period, devices, reads", [(1.0, [0, 1], 2), (0.0, [0] * 5, 5)])
def test_device_change_or_zero_period_reads_again(period, devices, reads):
    f = _Fake()
    m = IterStatsMemInfo(period, read=f.read, clock=f.clock)
    for device in devices:
        m(device)
    assert f.reads == reads
