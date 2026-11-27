# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GpuTimestamp keeps torch.cuda.Event's timing semantics (perf_metrics_manager).

Armed by arm_gpu_timestamps for a model with RuntimeHooks.gpu_timestamps.

The stamp comes from the native op (trtllm.globaltimer_stamp). Where the op is not built, these tests
stand in a test-local Triton kernel that does the same store, so the timestamp's stream and timing logic is still
covered; production code has no such fallback.
"""

import pytest
import torch
from cuda.bindings import driver as cu

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


# A fixed spin of tens of milliseconds; the tests compare against the span that actually ran
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
    # Keep the GPU busy while the host sets up s and e (first record creates a stream and pinned memory), so ts and
    # s execute back to back instead of ts firing on an idle GPU before s is even enqueued.
    _spin()
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
    stream = torch.cuda.Stream()
    # Warm record: allocates the follower stream and buffers, so the gated record below issues no host sync.
    t.record(stream)
    t.synchronize()
    assert t.query()
    flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    (err,) = cu.cuStreamWaitValue32(
        cu.CUstream(stream.cuda_stream),
        cu.CUdeviceptr(flag.data_ptr()),
        1,
        cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
    )
    if err == cu.CUresult.CUDA_ERROR_NOT_SUPPORTED:
        pytest.skip("this device cannot hold a stream on a host flag")
    assert err == cu.CUresult.CUDA_SUCCESS, err
    try:
        t.record(stream)
        assert not t.query()  # held behind the host flag
    finally:
        flag.numpy()[0] = 1
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


def test_kill_switch(monkeypatch):
    # Registered first, so the process-wide arming is restored however the test ends.
    monkeypatch.setattr(pmm, "_GPU_TIMESTAMPS", pmm._GPU_TIMESTAMPS)
    monkeypatch.setattr(pmm, "_STAMP_OP", pmm._STAMP_OP)
    monkeypatch.setenv("TRTLLM_DISABLE_GPU_TIMESTAMPS", "1")
    assert pmm.arm_gpu_timestamps() is False
    monkeypatch.delenv("TRTLLM_DISABLE_GPU_TIMESTAMPS")
    monkeypatch.setattr(pmm, "prefer_pinned", lambda: True)
    assert pmm.arm_gpu_timestamps() is True
    # Without pinned memory (Confidential Compute) the executor keeps timed CUDA events.
    monkeypatch.setattr(pmm, "prefer_pinned", lambda: False)
    assert pmm.arm_gpu_timestamps() is False


def test_a_capture_after_a_record_leaves_the_stamp_unrecorded():
    t = pmm.GpuTimestamp()
    t.record()
    t.synchronize()
    assert t._recorded
    x = torch.zeros(4, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x.add_(1)
        t.record()
    assert not t._recorded  # a stale stamp cannot pair in elapsed_time


def test_follower_runs_at_the_greatest_priority_and_is_reused():
    import gc

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart

    # The device's own range: torch.cuda.Stream.priority_range() reports PyTorch's
    # narrower pool range.
    err, _, greatest = cudart.cudaDeviceGetStreamPriorityRange()
    assert err == cudart.cudaError_t.cudaSuccess
    t = pmm.GpuTimestamp()
    t.record()
    t.synchronize()
    assert t._follower.priority == greatest
    handle = t._follower.cuda_stream
    del t
    gc.collect()
    u = pmm.GpuTimestamp()
    u.record()
    u.synchronize()
    assert u._follower.cuda_stream == handle  # taken from the pool, not a new stream


def test_stamps_stay_bounded_in_a_loop():
    import gc

    gc.collect()
    device = torch.cuda.current_device()
    before = len(pmm._FOLLOWER_POOL.get(device, []))
    for _ in range(20):
        t = pmm.GpuTimestamp()
        t.record()
        t.synchronize()
        del t
    # One stream circulates; at most one more than before was ever created.
    assert len(pmm._FOLLOWER_POOL.get(device, [])) <= before + 1


def test_two_records_in_a_row_increase():
    t = pmm.GpuTimestamp()
    t.record()
    t.synchronize()
    first = int(t._host[0])
    _spin()
    t.record()
    t.synchronize()
    assert int(t._host[0]) > first
