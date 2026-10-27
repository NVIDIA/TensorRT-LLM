# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GpuTimestamp keeps torch.cuda.Event's timing semantics (perf_metrics_manager; armed by arm_gpu_timestamps).

The stamp comes from the native op (trtllm.minimax_m31_globaltimer_stamp). Where the op is not built, these tests
stand in a test-local Triton kernel that does the same store, so the timestamp's stream and timing logic is still
covered; production code has no such fallback.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import perf_metrics_manager as pmm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


_STAND_IN = None


def _stand_in_stamp_op():
    """A test-only %globaltimer store with the native op's signature, for images without the op."""
    global _STAND_IN
    if _STAND_IN is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _globaltimer_test_kernel(out_ptr):
            tl.store(out_ptr, tl.extra.cuda.globaltimer())

        _STAND_IN = lambda out: _globaltimer_test_kernel[(1,)](out)  # noqa: E731
    return _STAND_IN


@pytest.fixture(autouse=True)
def _stamp_path(monkeypatch):
    monkeypatch.setattr(pmm, "_STAMP_OP", None)
    if pmm._native_stamp_op() is None:
        monkeypatch.setattr(pmm, "_native_stamp_op", _stand_in_stamp_op)


# A fixed spin of tens of milliseconds on Rubin-class clocks; the tests compare against the span that actually ran
# (timed events), so no cycles-per-ms calibration is needed (torch.cuda._sleep calibration is unreliable).
_SPIN_CYCLES = 40_000_000


def _spin() -> None:
    torch.cuda._sleep(_SPIN_CYCLES)


def test_elapsed_matches_timed_events():
    # Warm the spin and the stamp kernel first, so neither's first launch lands inside the measured window.
    _spin()
    warm = pmm.GpuTimestamp()
    warm.record()
    warm.synchronize()
    torch.cuda.synchronize()
    s, e = pmm.GpuTimestamp(), pmm.GpuTimestamp()
    ts, te = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts.record()
    s.record()
    _spin()
    e.record()
    te.record()
    torch.cuda.synchronize()
    ref = ts.elapsed_time(te)
    got = s.elapsed_time(e)
    assert ref > 1.0
    assert abs(got - ref) < 0.05 + 0.02 * ref, (got, ref)


def test_query_and_unrecorded_semantics():
    t = pmm.GpuTimestamp()
    assert t.query()  # unrecorded: True, like torch.cuda.Event
    with pytest.raises(RuntimeError):
        t.elapsed_time(pmm.GpuTimestamp())
    _spin()
    t.record()
    assert not t.query()  # behind the spin
    t.synchronize()
    assert t.query()


def test_record_under_capture_records_nothing():
    t = pmm.GpuTimestamp()
    x = torch.zeros(4, device="cuda")
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x.add_(1)  # warm the op outside capture
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        x.add_(1)
        t.record()
    assert not t._recorded
    g.replay()
    torch.cuda.synchronize()
    assert t.query()


def test_timing_event_follows_the_arm(monkeypatch):
    monkeypatch.setattr(pmm, "_GPU_TIMESTAMPS", False)
    assert isinstance(pmm.timing_event(), torch.cuda.Event)
    monkeypatch.setattr(pmm, "_GPU_TIMESTAMPS", True)
    assert isinstance(pmm.timing_event(), pmm.GpuTimestamp)


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_GPU_TIMESTAMPS", "1")
    assert pmm.arm_gpu_timestamps() is False
    monkeypatch.delenv("TRTLLM_M31_DISABLE_GPU_TIMESTAMPS")
    assert pmm.arm_gpu_timestamps() is True
    monkeypatch.setattr(pmm, "_GPU_TIMESTAMPS", False)
