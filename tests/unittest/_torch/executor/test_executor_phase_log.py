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
"""Per-rank executor phase log (TRTLLM_EXECUTOR_PHASE_LOG)."""

import pytest

from tensorrt_llm._torch.pyexecutor.executor_phase_log import (
    PHASE_LOG_ENV,
    ExecutorPhaseLog,
    phase_log_threshold_ms,
)


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += ms / 1e3


def _make(threshold_ms=5.0):
    clock = _Clock()
    lines = []
    log = ExecutorPhaseLog(threshold_ms, 3, lines.append, clock=clock, wall_clock=lambda: 1.5)
    return log, clock, lines


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("0", 0.0), ("2.5", 2.5)])
def test_threshold_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(PHASE_LOG_ENV, raising=False)
    else:
        monkeypatch.setenv(PHASE_LOG_ENV, value)
    assert phase_log_threshold_ms() == expected


@pytest.mark.parametrize("value", ["-1", "fast", "nan", "1,5"])
def test_threshold_ignores_an_invalid_value(monkeypatch, value):
    monkeypatch.setenv(PHASE_LOG_ENV, value)
    assert phase_log_threshold_ms() is None


def test_quiet_iteration_is_not_logged():
    log, clock, lines = _make()
    log.begin()
    clock.advance(1.0)
    log.mark("schedule")
    clock.advance(2.0)
    log.mark("forward")
    log.count("admitted", 0)
    assert log.end(7) is None
    assert lines == []


def test_request_event_logs_phase_durations():
    log, clock, lines = _make()
    log.begin()
    clock.advance(1.25)
    log.mark("schedule")
    log.count("admitted", 1)
    clock.advance(0.5)
    log.mark("forward")
    clock.advance(0.25)
    line = log.end(42)
    assert lines == [line]
    assert line.startswith("[phase] rank=3 iter=42 t0=1.500000 total=2.000 busy=2.000")
    assert "schedule=1.250" in line and "forward=0.500" in line
    assert line.endswith("admitted=1")


def test_busy_iteration_is_logged_without_events():
    log, clock, lines = _make(threshold_ms=5.0)
    log.begin()
    clock.advance(6.0)
    log.mark("schedule")
    line = log.end(1)
    assert line is not None and "busy=6.000" in line and "schedule=6.000" in line


def test_waiting_for_the_gpu_does_not_count_as_busy():
    log, clock, lines = _make(threshold_ms=5.0)
    log.begin()
    clock.advance(2.0)
    log.mark("forward")
    clock.advance(7.0)
    log.mark("wait_previous")
    clock.advance(1.0)
    assert log.end(1) is None
    log.begin()
    clock.advance(2.0)
    log.mark("forward")
    clock.advance(7.0)
    log.mark("wait_previous")
    clock.advance(3.5)
    line = log.end(2)
    assert line is not None and "total=12.500 busy=5.500" in line


def test_counts_accumulate_and_reset_between_iterations():
    log, clock, lines = _make()
    log.begin()
    log.count("first", 1)
    log.count("first", 1)
    log.count("finished", 1)
    assert log.end(1).endswith("first=2 finished=1")
    log.begin()
    clock.advance(1.0)
    assert log.end(2) is None
    assert len(lines) == 1


def test_mark_and_end_without_begin_are_ignored():
    log, clock, lines = _make()
    log.mark("schedule")
    assert log.end(1) is None
    log.begin()
    log.count("admitted", 1)
    assert log.end(2) is not None
    # end() closed the iteration: a stray mark/end is ignored again.
    log.mark("tail")
    assert log.end(3) is None
