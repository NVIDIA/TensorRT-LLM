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
    """Count nested time once and retain time outside measured phases."""
    clock, log = timing
    clock.side_effect = [0, 2, 3, 4, 6, 9, 12]
    with _startup._StartupTimer("executor") as timer:
        timer.mark_initialization("configuration")
        with timer.phase("model"):
            with timer.phase("weights"):
                pass
    assert timer.timings == {"configuration": 2, "model": 6}
    assert timer.depth == 0
    log.info.assert_any_call("[startup][pid=42] executor/weights: done in 2.000s")
    log.info.assert_any_call(
        "[startup][pid=42] executor: done, total=12.000s, unattributed=4.000s | "
        "configuration=2.000s, model=6.000s"
    )


def test_repeated_phases_accumulate(timing):
    """Keep both occurrences of a repeated phase in the summary."""
    clock, _ = timing
    clock.side_effect = [0, 1, 3, 4, 7, 8]
    with _startup._StartupTimer("executor") as timer:
        with timer.phase("allocation"):
            pass
        with timer.phase("allocation"):
            pass
    assert timer.timings == {"allocation": 5}


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failed_nested_phase_preserves_time_and_exception(timing, error_type):
    """A failed phase is timed once, labelled failed, and never suppresses errors."""
    clock, log = timing
    clock.side_effect = [0, 1, 2, 4, 6, 8]
    error = error_type("initialization failed")
    timer = _startup._StartupTimer("executor")
    with pytest.raises(error_type) as caught:
        with timer:
            with timer.phase("model"):
                with timer.phase("weights"):
                    raise error
    assert caught.value is error
    assert timer.depth == 0
    assert timer.timings == {"model": 5}
    messages = [c.args[0] for c in log.info.call_args_list]
    assert "[startup][pid=42] executor/weights: failed in 2.000s" in messages
    assert "[startup][pid=42] executor/model: failed in 5.000s" in messages
    assert not any(": done" in message for message in messages)
    assert any("executor: failed, total=8.000s, unattributed=3.000s" in m for m in messages)
