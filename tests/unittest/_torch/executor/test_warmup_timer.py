# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression coverage for process-local warmup timing."""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import warmup_timer

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def warmup_timing(monkeypatch):
    """Exercise the real timing helpers without constructing a GPU engine."""
    engine = warmup_timer._WarmupTimer(rank=3)
    engine.purpose = "memory_profiling"
    engine.pass_index = 1
    clock = Mock()
    log = Mock()
    monkeypatch.setattr(warmup_timer.time, "perf_counter", clock)
    monkeypatch.setattr(warmup_timer, "logger", log)
    monkeypatch.setattr(warmup_timer.os, "getpid", lambda: 42)
    return engine, clock, log


def test_warmup_phase_repeated_names(warmup_timing):
    """Repeated names accumulate and the summary reports identity and percentages."""
    engine, clock, log = warmup_timing
    clock.side_effect = [0, 2, 3, 6]
    for _ in range(2):
        with engine.phase("attention"):
            pass
    assert engine.timings == {"attention": 5}
    engine.summary(10)
    log.info.assert_any_call(
        "[warmup][pid=42][rank=3][purpose=memory_profiling][pass=1] "
        "summary: total=10.0s | attention=5.0s (50%)"
    )
    log.warning.assert_not_called()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_warmup_phase_failure_records_elapsed(warmup_timing, error_type):
    """Preserve elapsed time and the original exception on interrupted warmup."""
    engine, clock, log = warmup_timing
    clock.side_effect = [2, 7]
    error = error_type("warmup failed")
    with pytest.raises(error_type) as caught:
        with engine.phase("attention"):
            raise error
    assert caught.value is error
    assert engine.timings == {"attention": 5}
    log.warning.assert_called_once_with(
        "[warmup][pid=42][rank=3][purpose=memory_profiling][pass=1] attention: failed after 5.0s"
    )
    assert not any(": done" in c.args[0] for c in log.info.call_args_list)


@pytest.mark.parametrize(
    "timings, expected",
    [
        ({}, "summary: total=0.0s (no phases ran)"),
        ({"attention": 0.0}, "summary: total=0.0s | attention=0.0s (0%)"),
    ],
)
def test_warmup_summary_empty_or_zero(warmup_timing, timings, expected):
    """Empty or zero-duration summaries remain finite and do not warn."""
    engine, _, log = warmup_timing
    engine.timings = timings
    engine.summary(0)
    assert log.info.call_args.args[0].endswith(expected)
    log.warning.assert_not_called()


@pytest.mark.parametrize("over_threshold", [False, True])
def test_warmup_slow_phase_warning(warmup_timing, over_threshold):
    """Warn only when a phase strictly exceeds the slow-phase threshold."""
    engine, _, log = warmup_timing
    elapsed = engine._WARMUP_SLOW_PHASE_SEC + int(over_threshold)
    engine.timings = {"attention": elapsed}
    engine.summary(elapsed)
    if over_threshold:
        log.warning.assert_called_once_with(
            "[warmup][pid=42][rank=3][purpose=memory_profiling][pass=1] "
            f"slow phases (>{engine._WARMUP_SLOW_PHASE_SEC:.0f}s): attention={elapsed:.1f}s"
        )
    else:
        log.warning.assert_not_called()


def test_warmup_pass_reset_and_partial_summary(warmup_timing):
    """Each pass starts fresh and a failed pass still emits its own summary."""
    _, clock, log = warmup_timing
    clock.side_effect = [0, 1, 3, 4, 10, 11, 16, 20]
    timer = warmup_timer._WarmupTimer(rank=3)
    timer.purpose = "memory_profiling"
    with timer:
        with timer.phase("attention"):
            pass
    timer.purpose = "cuda_graph_capture"
    error = ValueError("failure")
    with pytest.raises(ValueError) as caught:
        with timer:
            with timer.phase("general"):
                raise error
    assert caught.value is error
    assert timer.pass_index == 2
    assert timer.timings == {"general": 5}
    log.info.assert_any_call(
        "[warmup][pid=42][rank=3][purpose=cuda_graph_capture][pass=2] "
        "summary: total=10.0s | general=5.0s (50%)"
    )


@pytest.mark.parametrize("log_start", [False, True])
def test_warmup_shapes_are_not_double_counted(warmup_timing, log_start):
    """Shape timing stays visible without inflating the enclosing phase total."""
    _, clock, log = warmup_timing
    clock.side_effect = [0, 1, 2, 5, 7, 10]
    with warmup_timer._WarmupTimer(rank=3) as timer:
        with timer.phase("general"):
            with timer.phase("general shape", record=False, log_start=log_start):
                pass
    assert timer.timings == {"general": 6}
    messages = [call.args[0] for call in log.info.call_args_list]
    assert any("general shape: done in 3.0s" in msg for msg in messages)
    assert any("general shape: start" in msg for msg in messages) == log_start
    assert messages[-1].endswith("general=6.0s (60%)")
