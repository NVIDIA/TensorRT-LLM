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
"""Tests for the opt-in small-block receive scatter (TRTLLM_KV_BOUNCE_RECV_SCATTER_TUNED).

The CPU tests check the switch, that only the receive scatter takes the small block, and that the
default calls are unchanged. The GPU test checks that the small block lands exactly the bytes of the
default block at mixed fragment sizes and destination layouts.
"""

import queue
import threading
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

needs_transport = pytest.mark.skipif(
    not _HAVE_TRANSPORT, reason="bounce impl import needs CUDA bindings"
)

_ENV = "TRTLLM_KV_BOUNCE_RECV_SCATTER_TUNED"
_MIB = 1024 * 1024

# ---------------------------------------------------------------------------
# knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected", [(None, False), ("", False), (" ", False), ("0", False), ("1", True)]
)
def test_recv_scatter_tuned_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    assert bcfg.recv_scatter_tuned_from_env() is expected


@pytest.mark.parametrize("raw", ["2", "true", "on", "-1"])
def test_recv_scatter_tuned_rejects_bad_value(monkeypatch, raw):
    monkeypatch.setenv(_ENV, raw)
    with pytest.raises(ValueError, match=_ENV):
        bcfg.recv_scatter_tuned_from_env()


@needs_transport
def test_create_bounce_bad_tuned_raises_instead_of_disabling(monkeypatch):
    monkeypatch.setenv(_ENV, "yes")

    def _no_build(*args, **kwargs):
        raise AssertionError("from_config must not run for a bad value")

    monkeypatch.setattr(btr.VmmBounceTransport, "from_config", classmethod(_no_build))
    with pytest.raises(ValueError, match=_ENV):
        btr.create_bounce(SimpleNamespace(), SimpleNamespace(), device_id=0, page_table=None)


@needs_transport
def test_small_block_is_8_kib_on_2_warps():
    assert bgs._RECV_SCATTER_BLOCK * bgs._ELEM_BYTES == 8192
    assert bgs._RECV_SCATTER_NUM_WARPS == 2
    # the default block of the gather and of the knob-off scatter is untouched
    assert (bgs._BLOCK, bgs._NUM_WARPS) == (4096, 4)


# ---------------------------------------------------------------------------
# launch shape: only the receive scatter with the switch on takes the small block
# ---------------------------------------------------------------------------


def _record_launches(monkeypatch):
    seen = []

    def launch(dst, src, sizes, stream, **kwargs):
        seen.append(kwargs)
        return True

    monkeypatch.setattr(bgs, "_launch_batched_copy", launch)
    return seen


_SIZES = np.array([4096, 8192], dtype=np.int64)
_OFFSETS = np.array([0, 4096], dtype=np.int64)
_PTRS = np.array([0x10000, 0x20000], dtype=np.int64)


@needs_transport
def test_scatter_default_call_is_unchanged(monkeypatch):
    seen = _record_launches(monkeypatch)
    bgs.scatter_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=11)
    assert seen == [{}]


@needs_transport
def test_scatter_tuned_passes_the_small_block(monkeypatch):
    seen = _record_launches(monkeypatch)
    bgs.scatter_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=11, tuned=True)
    assert seen == [{"block": 512, "num_warps": 2}]


@needs_transport
def test_gather_never_takes_the_small_block(monkeypatch):
    seen = _record_launches(monkeypatch)
    monkeypatch.setenv(_ENV, "1")
    bgs.gather_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=9)
    bgs.gather_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=9, alloc_on_stream=True)
    assert seen == [{"alloc_on_stream": False}, {"alloc_on_stream": True}]


@needs_transport
def test_scatter_tuned_falls_back_like_the_default(monkeypatch):
    """When the batched copy is unavailable, both modes take the same per-fragment loop."""
    monkeypatch.setattr(bgs, "_launch_batched_copy", lambda *a, **k: False)
    loops = []
    monkeypatch.setattr(
        bgs, "_copy_frags", lambda pairs, sizes, stream: loops.append((list(pairs), list(sizes)))
    )
    bgs.scatter_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=11)
    bgs.scatter_contiguous(0x900000, _PTRS, _SIZES, _OFFSETS, stream=11, tuned=True)
    assert len(loops) == 2 and loops[0] == loops[1]


class _FakeBuf:
    def numpy(self):
        return np.zeros(64, dtype=np.int64)

    def __getitem__(self, item):
        return self

    def copy_(self, other, non_blocking=False):
        return self


class _FakeKernel:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, kwargs))

        return launch


def _patch_kernel(monkeypatch):
    import contextlib

    kernel = _FakeKernel()
    fake_torch = SimpleNamespace(
        device=lambda *a: ("device",) + a,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            ExternalStream=lambda handle: ("ext", handle),
            stream=lambda s: contextlib.nullcontext(),
        ),
    )
    monkeypatch.setattr(bgs, "_HAVE_TRITON", True)
    monkeypatch.setattr(bgs, "torch", fake_torch, raising=False)
    monkeypatch.setattr(bgs, "triton", SimpleNamespace(cdiv=lambda a, b: -(-a // b)), raising=False)
    monkeypatch.setattr(bgs, "_batched_copy_kernel", kernel, raising=False)
    monkeypatch.setattr(bgs, "_get_meta_buffers", lambda h, need, dev: (_FakeBuf(), _FakeBuf()))
    return kernel


@needs_transport
def test_launch_grid_follows_the_block(monkeypatch):
    kernel = _patch_kernel(monkeypatch)
    # 8 KiB and 72 KiB fragments: 512 and 4608 elements
    sizes = np.array([8192, 73728], dtype=np.int64)
    bgs.scatter_contiguous(0x900000, _PTRS, sizes, np.array([0, 8192]), stream=11)
    bgs.scatter_contiguous(0x900000, _PTRS, sizes, np.array([0, 8192]), stream=11, tuned=True)
    assert kernel.launches[0] == ((2, 2), {"BLOCK": 4096, "num_warps": 4})
    assert kernel.launches[1] == ((2, 9), {"BLOCK": 512, "num_warps": 2})


# ---------------------------------------------------------------------------
# transport: the receive scatter forwards the switch
# ---------------------------------------------------------------------------


class _Alloc:
    """The SlotAllocator surface the constructor touches."""

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        self.capacity = capacity_bytes

    def reg_descs(self):
        return []


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a, **k):
        self.lines.append(msg)

    warning = error = info

    def info_once(self, msg, *a, **k):
        pass

    warning_once = info_once


def _transport(monkeypatch, tuned):
    if tuned is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, tuned)
    monkeypatch.setattr(btr, "SlotAllocator", _Alloc)
    monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self, mode=0: 0)
    monkeypatch.setattr(
        btr.VmmBounceTransport,
        "_start_scatter_worker",
        lambda self, name: setattr(self, "_scatter_q", queue.Queue()),
    )
    log = _Log()
    monkeypatch.setattr(btr, "logger", log)
    t = btr.VmmBounceTransport(
        SimpleNamespace(register_memory=lambda d: None),
        device_id=0,
        capacity_bytes=64 * _MIB,
        phys_chunk_size=32 * _MIB,
        block_bytes_per_group=[1024],
        min_blocks=1,
        direction="recv",
    )
    t._scatter_stream = 7
    t._stop = threading.Event()
    return t, log


def _patch_scatter(monkeypatch, strict):
    """Record each scatter call and each stream sync, in order.

    ``strict`` takes the knob-off signature, so any extra keyword fails the call.
    """
    calls = []
    if strict:

        def fake_scatter(src_base, dst_ptrs, sizes, offsets, *, stream):
            calls.append(("scatter", list(dst_ptrs), list(offsets), list(sizes), stream, None))

    else:

        def fake_scatter(src_base, dst_ptrs, sizes, offsets, *, stream, tuned):
            calls.append(("scatter", list(dst_ptrs), list(offsets), list(sizes), stream, tuned))

    monkeypatch.setattr(btr, "scatter_contiguous", fake_scatter)
    monkeypatch.setattr(btr, "CUASSERT", lambda ret: ret)
    monkeypatch.setattr(
        btr,
        "cudart",
        SimpleNamespace(cudaStreamSynchronize=lambda s: calls.append(("sync", s)) or (0,)),
    )
    return calls


_DESCS = [
    (0x100000, np.array([0x2000, 0x5000], dtype=np.int64), np.array([64, 128], dtype=np.int64)),
    (0x300000, np.array([0x9000], dtype=np.int64), np.array([256], dtype=np.int64)),
]
# what the one batched scatter receives: every writer's fragments, each read from its own source
_DSTS = [0x2000, 0x5000, 0x9000]
_SRCS = [0x100000, 0x100040, 0x300000]
_FRAGS = [64, 128, 256]


@needs_transport
@pytest.mark.parametrize("raw", [None, "0"])
def test_transport_off_keeps_the_default_scatter_call(monkeypatch, raw):
    t, log = _transport(monkeypatch, raw)
    calls = _patch_scatter(monkeypatch, strict=True)
    t._scatter_blocking(_DESCS)
    assert calls == [("scatter", _DSTS, _SRCS, _FRAGS, 7, None), ("sync", 7)]
    assert not any("tuned on" in m for m in log.lines)


@needs_transport
def test_transport_on_passes_tuned_to_the_receive_scatter(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    calls = _patch_scatter(monkeypatch, strict=False)
    t._scatter_blocking(_DESCS)
    # same fragments, same sources, same stream, one sync after the copy
    assert calls == [("scatter", _DSTS, _SRCS, _FRAGS, 7, True), ("sync", 7)]
    assert sum("[kv-bounce] recv scatter tuned on" in m for m in log.lines) == 1


# ---------------------------------------------------------------------------
# GPU: the small block lands the default block's exact bytes
# ---------------------------------------------------------------------------

# (fragment sizes in bytes, destination run length in fragments). Every size is a multiple of 16,
# so both modes run the batched kernel; sizes span below, at and above the 8 KiB program.
_GPU_CASES = {
    "pages_8k_r1": ([8192] * 3000, 1),
    "pages_8k_r7": ([8192] * 3000, 7),
    "mixed_r1": ([16, 48, 4096, 8192, 8208, 12288, 65536, 65552, 131072, 1 << 20] * 20, 1),
    "mixed_r3": ([16, 48, 4096, 8192, 8208, 12288, 65536, 65552, 131072, 1 << 20] * 20, 3),
    "small_r1": ([16, 32, 64, 128, 512, 1024, 2048] * 300, 1),
}


def _gpu_run(torch, sizes, run, tuned, seed):
    """Scatter ``sizes`` from one contiguous source to shuffled, disjoint destination slots.

    The slot order is kept in runs of ``run`` consecutive slots. Returns the destination pool.
    """
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes, dtype=np.int64)
    slot = int(-(-sizes.max() // 256) * 256)
    n = sizes.size
    nruns = -(-n // run)
    order = (rng.permutation(nruns)[:, None] * run + np.arange(run)[None, :]).reshape(-1)[:n]
    src = torch.randint(1, 256, (int(sizes.sum()),), dtype=torch.uint8, device="cuda")
    pool = torch.zeros(nruns * run * slot, dtype=torch.uint8, device="cuda")
    offsets = np.zeros(n, dtype=np.int64)
    np.cumsum(sizes[:-1], out=offsets[1:])
    dst = np.int64(pool.data_ptr()) + order.astype(np.int64) * slot
    stream = btr.CUASSERT(btr.cudart.cudaStreamCreate())[0]
    try:
        bgs.scatter_contiguous(src.data_ptr(), dst, sizes, offsets, stream=stream, tuned=tuned)
        btr.CUASSERT(btr.cudart.cudaStreamSynchronize(stream))
    finally:
        btr.CUASSERT(btr.cudart.cudaStreamDestroy(stream))
    host_src = src.cpu().numpy()
    expected = np.zeros(pool.numel(), dtype=np.uint8)
    for k in range(n):
        d = int(order[k]) * slot
        expected[d : d + sizes[k]] = host_src[offsets[k] : offsets[k] + sizes[k]]
    return pool.cpu().numpy(), expected


@needs_transport
@pytest.mark.parametrize("case", sorted(_GPU_CASES))
def test_gpu_tuned_scatter_is_bitwise_the_default(monkeypatch, case):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not bgs._HAVE_TRITON:
        pytest.skip("the batched copy kernel needs Triton")
    launches = []
    real_launch = bgs._launch_batched_copy

    def launch(*args, **kwargs):
        ran = real_launch(*args, **kwargs)
        launches.append((kwargs.get("block", bgs._BLOCK), ran))
        return ran

    monkeypatch.setattr(bgs, "_launch_batched_copy", launch)
    sizes, run = _GPU_CASES[case]
    torch.manual_seed(127)
    off, expected = _gpu_run(torch, sizes, run, tuned=False, seed=5)
    torch.manual_seed(127)
    on, expected_on = _gpu_run(torch, sizes, run, tuned=True, seed=5)
    assert np.array_equal(expected, expected_on)  # same source bytes and layout
    assert np.array_equal(off, expected)
    assert np.array_equal(on, off)
    # both ran the batched kernel, with the default and the small block
    assert launches == [(4096, True), (512, True)]
