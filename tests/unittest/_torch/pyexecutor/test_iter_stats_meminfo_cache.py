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


def test_device_change_reads_again():
    f = _Fake()
    m = IterStatsMemInfo(1.0, read=f.read, clock=f.clock)
    m(0)
    m(1)
    assert f.reads == 2


def test_zero_period_reads_every_call():
    f = _Fake()
    m = IterStatsMemInfo(0.0, read=f.read, clock=f.clock)
    for _ in range(5):
        m(0)
    assert f.reads == 5
