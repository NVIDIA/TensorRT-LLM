# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for startup timing and exception accounting."""

from unittest.mock import Mock

import pytest

from tensorrt_llm import _startup

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def timing(monkeypatch):
    """Replace wall time and logging with deterministic observations."""
    clock = Mock()
    log = Mock()
    monkeypatch.setattr(_startup.time, "perf_counter", clock)
    monkeypatch.setattr(_startup, "logger", log)
    monkeypatch.setattr(_startup.os, "getpid", lambda: 42)
    return clock, log


def test_nested_phases_and_unattributed_time(timing):
    """Export nested phases without double counting them in the summary."""
    clock, log = timing
    clock.side_effect = [0, 2, 3, 4, 6, 9, 12]
    metrics = {}
    with _startup._StartupTimer("executor") as timer:
        timer.mark_initialization("configuration")
        with timer.phase("model", metrics=metrics):
            with timer.phase("weights", metrics=metrics):
                pass
    assert timer.timings == {"configuration": 2, "model": 6}
    assert metrics == {"model": 6, "weights": 2}
    assert clock.call_count == 7
    assert timer.depth == 0
    log.info.assert_any_call("[startup][pid=42] executor/weights: done in 2.000s")
    log.info.assert_any_call(
        "[startup][pid=42] executor: done, total=12.000s, unattributed=4.000s | "
        "configuration=2.000s, model=6.000s"
    )


def test_repeated_phases_accumulate(timing):
    """Accumulate repeated phases in both the summary and exported metrics."""
    clock, _ = timing
    clock.side_effect = [0, 1, 3, 4, 7, 8]
    metrics = {}
    with _startup._StartupTimer("executor") as timer:
        with timer.phase("allocation", metrics=metrics, metric_name="allocation_seconds"):
            pass
        with timer.phase("allocation", metrics=metrics, metric_name="allocation_seconds"):
            pass
    assert timer.timings == {"allocation": 5}
    assert metrics == {"allocation_seconds": 5}
    assert clock.call_count == 6


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failed_nested_phase_preserves_time_and_exception(timing, error_type):
    """A failed phase is timed once, labelled failed, and never suppresses errors."""
    clock, log = timing
    clock.side_effect = [0, 1, 2, 4, 6, 8]
    error = error_type("initialization failed")
    timer = _startup._StartupTimer("executor")
    metrics = {}
    with pytest.raises(error_type) as caught:
        with timer:
            with timer.phase("model", metrics=metrics):
                with timer.phase("weights", metrics=metrics):
                    raise error
    assert caught.value is error
    assert timer.depth == 0
    assert timer.timings == {"model": 5}
    assert metrics == {"model": 5, "weights": 2}
    messages = [c.args[0] for c in log.info.call_args_list]
    assert "[startup][pid=42] executor/weights: failed in 2.000s" in messages
    assert "[startup][pid=42] executor/model: failed in 5.000s" in messages
    assert not any(": done" in message for message in messages)
    assert any("executor: failed, total=8.000s, unattributed=3.000s" in m for m in messages)
