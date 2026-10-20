# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_HANG_CHECKPOINT_MIN_S: HangDetector.checkpoint() reschedules its timer at most once per interval
(flag off: on every call), and a hang is still detected once the checkpoints stop (no GPU)."""

import time

import pytest

from tensorrt_llm._torch.pyexecutor import hang_detector as hang_detector_module
from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector

pytestmark = pytest.mark.cpu_only


def _count_reschedules(monkeypatch):
    calls = []
    real = hang_detector_module.asyncio.run_coroutine_threadsafe

    def counting(coro, loop):
        calls.append(time.monotonic())
        return real(coro, loop)

    monkeypatch.setattr(hang_detector_module.asyncio, "run_coroutine_threadsafe", counting)
    return calls


@pytest.mark.parametrize("min_s, expected", [(None, 200), ("0.5", 1)])
def test_checkpoint_reschedules(monkeypatch, min_s, expected):
    if min_s is None:
        monkeypatch.delenv(hang_detector_module.HANG_CHECKPOINT_MIN_S_ENV, raising=False)
    else:
        monkeypatch.setenv(hang_detector_module.HANG_CHECKPOINT_MIN_S_ENV, min_s)
    calls = _count_reschedules(monkeypatch)
    hd = HangDetector(timeout=60)
    with hd:
        for _ in range(200):
            hd.checkpoint()
        assert len(calls) == expected
        assert not hd.detected()


def test_throttled_detector_still_fires_and_resumes_after_pause(monkeypatch):
    monkeypatch.setenv(hang_detector_module.HANG_CHECKPOINT_MIN_S_ENV, "0.2")
    calls = _count_reschedules(monkeypatch)
    fired = []
    hd = HangDetector(timeout=1, on_detected=lambda: fired.append(time.monotonic()))
    with hd:
        start = time.monotonic()
        while time.monotonic() - start < 0.5:  # steady progress: the timer is rescheduled a few times
            hd.checkpoint()
            time.sleep(0.01)
        assert 2 <= len(calls) <= 4 and not hd.detected()
        with hd.pause():  # pause cancels the timer; leaving it reschedules at once (not throttled)
            pass
        assert len(calls) >= 3
        last = time.monotonic()
        deadline = last + 3.0
        while time.monotonic() < deadline and not hd.detected():
            time.sleep(0.05)
        assert hd.detected() and fired
        assert fired[0] - last < 1.0 + 0.2 + 0.5  # timeout + interval + scheduling slack
