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
"""Real bounce copies must progress independently of unrelated legacy-default-stream work."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cuda.bindings import driver
from cuda.bindings import runtime as cudart

from tensorrt_llm._torch.disaggregation.native.bounce import gather_scatter
from tensorrt_llm._torch.disaggregation.native.bounce.core import TransferContext, TransferState
from tensorrt_llm._torch.disaggregation.native.bounce.impl import VmmBounceTransport
from tensorrt_llm._utils import CUASSERT

_WATCHDOG_SECONDS = 120.0


@contextmanager
def _held_legacy_default():
    flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    flag_host = flag.numpy()
    legacy = cudart.cudaStream_t(cudart.cudaStreamLegacy)
    done = CUASSERT(cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming))[0]
    fired = threading.Event()

    def release():
        flag_host[0] = 1

    def rescue():
        fired.set()
        release()

    watchdog = threading.Timer(_WATCHDOG_SECONDS, rescue)
    try:
        # Materialize the event before holding the stream, including its first record.
        CUASSERT(cudart.cudaEventRecord(done, legacy))
        CUASSERT(cudart.cudaEventSynchronize(done))
        (err,) = driver.cuStreamWaitValue32(
            driver.CUstream(driver.CU_STREAM_LEGACY),
            driver.CUdeviceptr(flag.data_ptr()),
            1,
            driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
        )
        if err == driver.CUresult.CUDA_ERROR_NOT_SUPPORTED:
            pytest.skip("device does not support a pinned-host stream wait")
        assert err == driver.CUresult.CUDA_SUCCESS, err
        watchdog.start()
        CUASSERT(cudart.cudaEventRecord(done, legacy))
        _assert_unrelated_pending(done, fired)
        yield done, fired
    finally:
        release()
        watchdog.cancel()
        if watchdog.ident is not None:
            watchdog.join()
        CUASSERT(cudart.cudaStreamSynchronize(legacy))
        CUASSERT(cudart.cudaEventDestroy(done))
    assert not fired.is_set(), "cleanup watchdog released the unrelated default stream"


def _assert_unrelated_pending(done, fired):
    (status,) = cudart.cudaEventQuery(done)
    assert status == cudart.cudaError_t.cudaErrorNotReady, (
        "bounce copy waited for unrelated legacy-default-stream work; "
        f"completion={status}, cleanup_watchdog_fired={fired.is_set()}"
    )


class _Slots:
    """Replace fabric allocation only; reservation state and settlement remain real."""

    def release(self, slot_id):
        pass

    def reclaim_expired(self):
        pass


@pytest.fixture
def small_transport():
    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA device")
    transport = VmmBounceTransport.__new__(VmmBounceTransport)
    transport._device_id = torch.cuda.current_device()
    transport._recv_alloc = _Slots()
    transport._init_recv_state()
    transport._send_stream = transport._new_stream()
    transport._send_stream_lock = threading.Lock()
    transport._scatter_stream = None
    transport._scatter_thread = None
    try:
        transport._start_scatter_worker("bounce-stream-contract")
        yield transport
    finally:
        if transport._scatter_thread is not None:
            transport._stop.set()
            transport._scatter_q.put(None)
            transport._scatter_thread.join(timeout=_WATCHDOG_SECONDS)
            assert not transport._scatter_thread.is_alive(), "scatter worker did not stop"
        for stream in (transport._send_stream, transport._scatter_stream):
            if stream is not None:
                CUASSERT(cudart.cudaStreamSynchronize(stream))
                with gather_scatter._meta_lock:
                    gather_scatter._meta_buffers.pop(int(stream), None)
                CUASSERT(cudart.cudaStreamDestroy(stream))


def _scatter(transport, bounce, dst_ptrs, sizes, rid):
    ctx = TransferContext(
        rid_slice=(rid, 0),
        slot_id=rid,
        base_addr=bounce.data_ptr(),
        per_writer_bytes=bounce.numel(),
        num_writers=1,
    )
    with transport._reserved_map_lock:
        transport._reserved_map[ctx.rid_slice] = ctx
    completed = threading.Event()
    results = []

    def on_done(ok):
        results.append(ok)
        completed.set()

    transport.record_result(ctx.rid_slice, 0, dst_ptrs, sizes, bounce.data_ptr(), on_done=on_done)
    # This bounds a lost callback; the earlier gate watchdog can always unblock a bad stream.
    assert completed.wait(2 * _WATCHDOG_SECONDS), "scatter completion callback never arrived"
    assert results == [True]
    assert ctx.state is TransferState.COMPLETED
    assert not transport.is_bounced(ctx.rid_slice)


@pytest.mark.parametrize(
    "fragment_sizes", [(32, 48, 16), (19, 31, 7)], ids=["aligned", "unaligned"]
)
def test_bounce_roundtrip_ignores_unrelated_default_work(small_transport, fragment_sizes):
    transport = small_transport
    sizes = np.array(fragment_sizes, dtype=np.int64)
    sources = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in fragment_sizes]
    destinations = [torch.empty_like(source) for source in sources]
    bounce = torch.empty(sum(fragment_sizes), dtype=torch.uint8, device="cuda")
    src_ptrs = np.array([source.data_ptr() for source in sources], dtype=np.int64)
    dst_ptrs = np.array([dest.data_ptr() for dest in destinations], dtype=np.int64)
    metadata = SimpleNamespace(src_ptrs=src_ptrs, dst_ptrs=dst_ptrs, sizes=sizes)
    producer = torch.cuda.Stream()
    reader = torch.cuda.Stream()
    ready = torch.cuda.Event()
    inputs = [torch.empty(n, dtype=torch.uint8, pin_memory=True) for n in fragment_sizes]
    readbacks = [torch.empty_like(value, pin_memory=True) for value in inputs]

    def produce(generation):
        for fragment, value in enumerate(inputs):
            value.numpy()[:] = np.arange(value.numel(), dtype=np.uint8) + fragment * 53 + generation
        with torch.cuda.stream(producer):
            for source, value in zip(sources, inputs, strict=True):
                source.copy_(value, non_blocking=True)
            ready.record()
        # The caller hands bounce ready data; it does not pass bounce an outstanding event.
        ready.synchronize()

    def check_bytes():
        with torch.cuda.stream(reader):
            for output, value in zip(destinations, readbacks, strict=True):
                value.copy_(output, non_blocking=True)
        reader.synchronize()
        for expected, actual in zip(inputs, readbacks, strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    # Prewarm the same gather, scatter worker, metadata buffers, producer and readback paths.
    produce(0)
    transport._gather_blocking(bounce.data_ptr(), metadata, bounce.numel())
    _scatter(transport, bounce, dst_ptrs, sizes, rid=0)
    check_bytes()
    produce(1)

    with _held_legacy_default() as (default_done, watchdog_fired):
        transport._gather_blocking(bounce.data_ptr(), metadata, bounce.numel())
        _assert_unrelated_pending(default_done, watchdog_fired)
        _scatter(transport, bounce, dst_ptrs, sizes, rid=1)
        _assert_unrelated_pending(default_done, watchdog_fired)
        # Read only on an independent stream after the real scatter completion callback.
        check_bytes()
        _assert_unrelated_pending(default_done, watchdog_fired)
