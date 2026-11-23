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
"""CPU-only tests for the opt-in KV bounce send-stream mode.

TRTLLM_KV_BOUNCE_SEND_STREAM_NONBLOCKING selects how the sender's gather stream is created. The CUDA
runtime and torch.cuda are monkeypatched.
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg

# impl and gather_scatter pull the CUDA bindings at import; skip when they are absent. Catch only
# ImportError so a genuine bug in the module still fails.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce import gather_scatter as bgs
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr

    _HAVE_TRANSPORT = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_TRANSPORT = False

_ENV = "TRTLLM_KV_BOUNCE_SEND_STREAM_NONBLOCKING"
_MIB = 1024 * 1024
_NON_BLOCKING_FLAG = 1  # value of cudaStreamNonBlocking in the CUDA runtime
_LEAST_PRIORITY = 0
_GREATEST_PRIORITY = -5

needs_transport = pytest.mark.skipif(
    not _HAVE_TRANSPORT, reason="bounce impl import needs CUDA bindings"
)


# ---------------------------------------------------------------------------
# (a) env parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 0),
        ("", 0),
        ("0", 0),
        ("1", 1),
        ("2", 2),
        (" 2 ", 2),
    ],
)
def test_send_stream_mode_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    assert bcfg.send_stream_mode_from_env() == expected


@pytest.mark.parametrize("raw", ["3", "-1", "yes", "1.0"])
def test_send_stream_mode_rejects_bad_value(monkeypatch, raw):
    monkeypatch.setenv(_ENV, raw)
    with pytest.raises(ValueError, match=_ENV):
        bcfg.send_stream_mode_from_env()


# ---------------------------------------------------------------------------
# (b) stream creation per mode
# ---------------------------------------------------------------------------


class _FakeCudart:
    """Records the stream and event calls; every call succeeds."""

    cudaStreamNonBlocking = _NON_BLOCKING_FLAG

    def __init__(self):
        self.calls = []
        self._next = 100

    def _handle(self):
        self._next += 1
        return self._next

    def cudaStreamCreate(self):
        self.calls.append(("cudaStreamCreate",))
        return (0, self._handle())

    def cudaStreamCreateWithFlags(self, flags):
        self.calls.append(("cudaStreamCreateWithFlags", flags))
        return (0, self._handle())

    def cudaDeviceGetStreamPriorityRange(self):
        self.calls.append(("cudaDeviceGetStreamPriorityRange",))
        return (0, _LEAST_PRIORITY, _GREATEST_PRIORITY)

    def cudaStreamCreateWithPriority(self, flags, priority):
        self.calls.append(("cudaStreamCreateWithPriority", flags, priority))
        return (0, self._handle())

    def cudaEventCreate(self):
        return (0, "event")

    def cudaEventRecord(self, event, stream):
        self.calls.append(("cudaEventRecord", stream))
        return (0,)

    def cudaEventSynchronize(self, event):
        return (0,)

    def cudaEventDestroy(self, event):
        return (0,)


class _FakeAlloc:
    """Stand-in for SlotAllocator: no device memory."""

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        self.capacity = capacity_bytes

    def reg_descs(self):
        return []


def _fake_cuassert(ret):
    assert ret[0] == 0
    return ret[1:] if len(ret) > 1 else None


def _make_transport(monkeypatch, fake_cudart):
    monkeypatch.setattr(btr, "cudart", fake_cudart)
    monkeypatch.setattr(btr, "CUASSERT", _fake_cuassert)
    monkeypatch.setattr(btr, "SlotAllocator", _FakeAlloc)
    # the real scatter worker starts (it creates the scatter stream); its loop exits at once
    monkeypatch.setattr(btr.VmmBounceTransport, "_scatter_loop", lambda self: None)
    agent = SimpleNamespace(register_memory=lambda d: None)
    t = btr.VmmBounceTransport(
        agent,
        device_id=0,
        capacity_bytes=64 * _MIB,
        phys_chunk_size=32 * _MIB,
        block_bytes_per_group=[1024],
    )
    t._scatter_thread.join(timeout=5)
    return t


_EXPECTED_SEND_CALLS = {
    0: [("cudaStreamCreate",)],
    1: [("cudaStreamCreateWithFlags", _NON_BLOCKING_FLAG)],
    2: [
        ("cudaDeviceGetStreamPriorityRange",),
        ("cudaStreamCreateWithPriority", _NON_BLOCKING_FLAG, _GREATEST_PRIORITY),
    ],
}


@needs_transport
@pytest.mark.parametrize("raw, mode", [(None, 0), ("0", 0), ("1", 1), ("2", 2)])
def test_transport_send_and_scatter_streams(monkeypatch, raw, mode):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    fake = _FakeCudart()
    t = _make_transport(monkeypatch, fake)
    assert t._send_stream_mode == mode
    # the send stream is created first, then the scatter stream, which is always blocking
    assert fake.calls == _EXPECTED_SEND_CALLS[mode] + [("cudaStreamCreate",)]
    assert t._send_stream != t._scatter_stream


@needs_transport
@pytest.mark.parametrize("raw, logged", [(None, False), ("0", False), ("1", True), ("2", True)])
def test_transport_logs_the_send_stream_mode_only_when_opted_in(monkeypatch, raw, logged):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    lines = []
    monkeypatch.setattr(btr.logger, "info", lambda msg, *a, **k: lines.append(msg))
    _make_transport(monkeypatch, _FakeCudart())
    mode_lines = [line for line in lines if "send stream mode" in line]
    assert mode_lines == ([f"[kv-bounce] send stream mode={raw}"] if logged else [])


@needs_transport
def test_transport_rejects_bad_mode(monkeypatch):
    monkeypatch.setenv(_ENV, "7")
    fake = _FakeCudart()
    with pytest.raises(ValueError, match=_ENV):
        _make_transport(monkeypatch, fake)
    assert fake.calls == []  # no stream created before the check


@needs_transport
def test_transport_bad_mode_allocates_and_registers_nothing(monkeypatch):
    monkeypatch.setenv(_ENV, "yes")
    fake = _FakeCudart()
    monkeypatch.setattr(btr, "cudart", fake)
    monkeypatch.setattr(btr, "CUASSERT", _fake_cuassert)
    allocs, registered = [], []

    class _RecordingAlloc(_FakeAlloc):
        def __init__(self, *args, **kwargs):
            allocs.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(btr, "SlotAllocator", _RecordingAlloc)
    agent = SimpleNamespace(register_memory=registered.append)
    with pytest.raises(ValueError, match=_ENV):
        btr.VmmBounceTransport(
            agent,
            device_id=0,
            capacity_bytes=64 * _MIB,
            phys_chunk_size=32 * _MIB,
            block_bytes_per_group=[1024],
        )
    assert allocs == [] and registered == [] and fake.calls == []


@needs_transport
def test_create_bounce_bad_mode_raises_instead_of_disabling(monkeypatch):
    # create_bounce turns any construction error into the disabled transport; a bad mode must
    # escape that, so a mistyped value cannot silently drop to the per-fragment path
    monkeypatch.setenv(_ENV, "true")

    def _no_build(*args, **kwargs):
        raise AssertionError("from_config must not run for a bad mode")

    monkeypatch.setattr(btr.VmmBounceTransport, "from_config", classmethod(_no_build))
    with pytest.raises(ValueError, match=_ENV):
        btr.create_bounce(SimpleNamespace(), SimpleNamespace(), device_id=0, page_table=None)


@needs_transport
def test_create_bounce_without_config_ignores_mode(monkeypatch):
    monkeypatch.setenv(_ENV, "true")
    assert isinstance(
        btr.create_bounce(None, None, device_id=0, page_table=None), btr.NoBounceTransport
    )


@needs_transport
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_new_stream_per_mode(monkeypatch, mode):
    fake = _FakeCudart()
    monkeypatch.setattr(btr, "cudart", fake)
    monkeypatch.setattr(btr, "CUASSERT", _fake_cuassert)
    btr.VmmBounceTransport._new_stream(object(), mode)
    assert fake.calls == _EXPECTED_SEND_CALLS[mode]


@needs_transport
def test_new_stream_default_is_blocking(monkeypatch):
    fake = _FakeCudart()
    monkeypatch.setattr(btr, "cudart", fake)
    monkeypatch.setattr(btr, "CUASSERT", _fake_cuassert)
    btr.VmmBounceTransport._new_stream(object())
    assert fake.calls == [("cudaStreamCreate",)]


@needs_transport
@pytest.mark.parametrize("raw, alloc_on_stream", [(None, False), ("1", True), ("2", True)])
def test_gather_blocking_forwards_alloc_on_stream(monkeypatch, raw, alloc_on_stream):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    fake = _FakeCudart()
    t = _make_transport(monkeypatch, fake)
    seen = []

    def fake_gather(dst_base, src_ptrs, sizes, offsets, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(btr, "gather_contiguous", fake_gather)
    meta = SimpleNamespace(
        src_ptrs=np.array([0x1000], dtype=np.int64),
        dst_ptrs=np.array([0x2000], dtype=np.int64),
        sizes=np.array([4096], dtype=np.int64),
    )
    t._gather_blocking(0x9000, meta, 4096)
    assert seen == [{"stream": t._send_stream, "alloc_on_stream": alloc_on_stream}]
    # the gather is still fenced by an event on the send stream
    assert ("cudaEventRecord", t._send_stream) in fake.calls


# ---------------------------------------------------------------------------
# (c) metadata allocation site in the batched copy
# ---------------------------------------------------------------------------


class _FakeBuf:
    def numpy(self):
        return np.zeros(64, dtype=np.int64)

    def __getitem__(self, item):
        return self

    def copy_(self, other, non_blocking=False):
        return self


class _FakeTorch:
    """Tracks which stream is current while _get_meta_buffers runs."""

    def __init__(self):
        self.current = None
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            ExternalStream=lambda handle: ("ext", handle),
            stream=self._stream_ctx,
        )

    @staticmethod
    def device(*args):
        return ("device",) + args

    @contextlib.contextmanager
    def _stream_ctx(self, stream):
        prev, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = prev


class _FakeKernel:
    def __init__(self):
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches += 1

        return launch


def _patch_batched_copy(monkeypatch):
    fake_torch = _FakeTorch()
    kernel = _FakeKernel()
    alloc_streams = []

    def fake_get_meta_buffers(stream_handle, need, dev):
        alloc_streams.append(fake_torch.current)
        return _FakeBuf(), _FakeBuf()

    monkeypatch.setattr(bgs, "_HAVE_TRITON", True)
    monkeypatch.setattr(bgs, "torch", fake_torch, raising=False)
    monkeypatch.setattr(bgs, "triton", SimpleNamespace(cdiv=lambda a, b: -(-a // b)), raising=False)
    monkeypatch.setattr(bgs, "_batched_copy_kernel", kernel, raising=False)
    monkeypatch.setattr(bgs, "_get_meta_buffers", fake_get_meta_buffers)
    return alloc_streams, kernel


_SIZES = np.array([4096, 8192], dtype=np.int64)
_OFFSETS = np.array([0, 4096], dtype=np.int64)
_PTRS = np.array([0x10000, 0x20000], dtype=np.int64)


@needs_transport
@pytest.mark.parametrize("alloc_on_stream", [False, True])
def test_launch_batched_copy_alloc_site(monkeypatch, alloc_on_stream):
    alloc_streams, kernel = _patch_batched_copy(monkeypatch)
    ok = bgs._launch_batched_copy(
        _PTRS, _PTRS + 0x100000, _SIZES, 7, alloc_on_stream=alloc_on_stream
    )
    assert ok and kernel.launches == 1
    assert alloc_streams == [("ext", 7) if alloc_on_stream else None]


@needs_transport
@pytest.mark.parametrize("alloc_on_stream", [False, True])
def test_gather_contiguous_forwards_alloc_on_stream(monkeypatch, alloc_on_stream):
    alloc_streams, kernel = _patch_batched_copy(monkeypatch)
    bgs.gather_contiguous(
        0x900000, _PTRS, _SIZES, _OFFSETS, stream=9, alloc_on_stream=alloc_on_stream
    )
    assert kernel.launches == 1
    assert alloc_streams == [("ext", 9) if alloc_on_stream else None]


@needs_transport
def test_gather_and_scatter_default_keep_legacy_alloc_site(monkeypatch):
    alloc_streams, kernel = _patch_batched_copy(monkeypatch)
    bgs.gather_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=9)
    bgs.scatter_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=11)
    assert kernel.launches == 2
    assert alloc_streams == [None, None]
