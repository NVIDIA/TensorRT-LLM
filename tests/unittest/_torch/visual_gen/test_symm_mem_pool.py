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
"""CPU tests for the symmetric-memory pool (no GPU).

The layout arithmetic is checked against a pure-Python reference, and the probe and ``reserve``
against a gloo group that cannot carry the pool. Everything behind the allocation runs on a CPU
stand-in for ``torch.distributed._symmetric_memory`` (``_FakeSymmMem``) whose blocks are files:
ranks are spawned gloo processes (they rendezvous through a file, no port) that map each
other's buffers, so the collectives, the layout agreed across ranks, the peer views and the
failure agreement are the pool's real code. What only a device can show (torch's own
rendezvous, the signal kernels, CUDA-graph capture) is not covered here.
"""

import inspect
import os
import re
import time
import weakref
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.parallel import symm_mem_pool
from tensorrt_llm._torch.visual_gen.parallel.symm_mem_pool import (
    DEFAULT_ALIGN,
    DEFAULT_TIMEOUT_MS,
    PoolLayout,
    SymmMemPool,
    probe_symmetric_memory,
)

pytestmark = pytest.mark.cpu_only


# =============================================================================
# Layout arithmetic
# =============================================================================


def _ref_region_nbytes(slot_nbytes: int, world_size: int, align: int) -> int:
    """Pure-Python reference: ceil(slot / world) rounded up to ``align``."""
    per_rank = (slot_nbytes + world_size - 1) // world_size
    return ((per_rank + align - 1) // align) * align


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
@pytest.mark.parametrize("slots", [1, 2, 3])
@pytest.mark.parametrize(
    "slot_nbytes",
    [1, 255, 256, 257, 4096, 5 * 256 + 3, 151_200 * 5120 + 151_200 * 10 * 4],
)
def test_pool_layout_matches_reference(world_size, slots, slot_nbytes):
    lay = PoolLayout.for_request(slot_nbytes, slots, world_size)
    region = _ref_region_nbytes(slot_nbytes, world_size, DEFAULT_ALIGN)
    assert lay.align == DEFAULT_ALIGN
    assert lay.region_nbytes == region
    assert lay.region_nbytes % DEFAULT_ALIGN == 0
    assert lay.slot_nbytes == world_size * region
    assert lay.slot_nbytes >= slot_nbytes
    assert lay.slot_nbytes - slot_nbytes < world_size * DEFAULT_ALIGN  # smallest such layout
    assert lay.nbytes == slots * lay.slot_nbytes
    # Regions tile the allocation in (slot, rank) order without gaps or overlaps, all aligned.
    starts = [lay.region_start(s, r) for s in range(slots) for r in range(world_size)]
    assert starts == [i * region for i in range(slots * world_size)]
    assert all(start % DEFAULT_ALIGN == 0 for start in starts)
    # A span inside the last region of the last slot ends inside the allocation.
    start, stop = lay.region_span(slots - 1, world_size - 1, 16, 8)
    assert (start, stop) == (lay.nbytes - region + 16, lay.nbytes - region + 24)
    assert lay.region_span(0, 0, 0, region) == (0, region)


def test_pool_layout_custom_align():
    lay = PoolLayout.for_request(1000, 2, 3, align=512)
    assert lay.region_nbytes == 512 and lay.slot_nbytes == 1536 and lay.nbytes == 3072
    assert lay.align == 512
    assert lay.region_start(1, 2) == 1536 + 1024


def test_pool_layout_rejects_out_of_range():
    lay = PoolLayout(4, 2, 512)
    with pytest.raises(ValueError, match="slot 2 out of range"):
        lay.region_span(2, 0, 0, 1)
    with pytest.raises(ValueError, match="source rank 4 out of range"):
        lay.region_span(0, 4, 0, 1)
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, 512, 1)  # starts past the region
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, 500, 13)  # ends past the region
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, 0, 0)  # empty
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, -1, 1)  # negative offset
    with pytest.raises(ValueError, match="positive multiple of align"):
        PoolLayout(4, 2, 500)
    with pytest.raises(ValueError, match="slots >= 1"):
        PoolLayout(4, 0, 512)
    with pytest.raises(ValueError, match="world_size >= 1"):
        PoolLayout(0, 1, 512)
    with pytest.raises(ValueError, match="positive multiple of align"):
        PoolLayout(4, 2, 0)
    with pytest.raises(ValueError, match="power of two"):
        PoolLayout(4, 1, 512, align=100)
    with pytest.raises(ValueError, match="power of two >= 16"):
        PoolLayout(4, 1, 512, align=8)
    with pytest.raises(ValueError, match="slot_nbytes >= 1"):
        PoolLayout.for_request(0, 1, 4)


def test_pool_layout_grow_only_merge():
    a = PoolLayout(4, 2, 1024)
    b = PoolLayout(4, 3, 512)
    assert a.covers(a) and not a.covers(b) and not b.covers(a)
    merged = a.merged(b)
    assert merged == PoolLayout(4, 3, 1024)
    assert merged.covers(a) and merged.covers(b)
    assert a.merged(PoolLayout(4, 1, 256)) == a  # a smaller request never shrinks the pool
    assert a.merged(PoolLayout(4, 1, 2048)) == PoolLayout(4, 2, 2048)  # larger regions only
    with pytest.raises(ValueError, match="different pools"):
        a.merged(PoolLayout(2, 2, 1024))
    with pytest.raises(ValueError, match="different pools"):
        a.merged(PoolLayout(4, 2, 1024, align=512))
    assert not a.covers(PoolLayout(4, 2, 1024, align=512))
    assert not a.covers(PoolLayout(2, 2, 1024))  # another world size


# =============================================================================
# Probe and reserve on a gloo group (one rank, in-process)
# =============================================================================


@pytest.fixture
def gloo_world():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def test_probe_declines_gloo_without_cuda(gloo_world):
    ok, reason = probe_symmetric_memory(gloo_world, torch.device("cpu"))
    assert ok is False
    assert reason.startswith("rank 0: ")
    assert "needs >= 2 ranks" in reason
    assert "device cpu is not a CUDA device" in reason
    assert "ProcessGroupGloo" in reason and "ProcessGroupNCCL" in reason
    # The same verdict with a string device and an explicit channel need.
    assert probe_symmetric_memory(gloo_world, "cpu", min_channels=4)[0] is False
    with pytest.raises(ValueError, match="process group"):
        probe_symmetric_memory(None, "cpu")


def test_probe_declines_a_non_positive_channel_request_before_allocating(
    gloo_world, fake_symm, collectives
):
    assert probe_symmetric_memory(gloo_world, "cpu", min_channels=0) == (
        False,
        "rank 0: min_channels must be >= 1 (got 0)",
    )
    # A value that cannot be compared is this rank's verdict too, not a raise of its own.
    ok, reason = probe_symmetric_memory(gloo_world, "cpu", min_channels=None)
    assert ok is False and "could not be evaluated: TypeError" in reason
    # Declined in the first agreement each time (the peers are in it), and nothing allocated.
    assert collectives == ["all_gather_object"] * 2
    assert fake_symm.allocated == 0


def test_another_symmetric_memory_backend_is_a_failed_precondition(gloo_world, monkeypatch):
    """The pool's channels are the CUDA backend's: one flag per channel and rank on the block's
    own pad. A process that selected another backend must fall back, on every rank."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    backend = {"name": "NVSHMEM"}
    monkeypatch.setattr(
        symm_mem_pool, "_symm_mem", lambda: SimpleNamespace(get_backend=lambda d: backend["name"])
    )
    reasons = symm_mem_pool._local_preconditions(gloo_world, torch.device("cuda:0"))
    assert any("symmetric-memory backend for cuda devices is 'NVSHMEM'" in r for r in reasons)
    backend["name"] = "CUDA"
    reasons = symm_mem_pool._local_preconditions(gloo_world, torch.device("cuda:0"))
    assert not any("symmetric-memory backend" in r for r in reasons)

    # A build without the module: reported once, as the import failure.
    def unimportable():
        raise ImportError("no symmetric memory in this build")

    monkeypatch.setattr(symm_mem_pool, "_symm_mem", unimportable)
    reasons = symm_mem_pool._local_preconditions(gloo_world, torch.device("cuda:0"))
    assert "is not importable (no symmetric memory in this build)" in reasons[0]
    assert not any("symmetric-memory backend" in r for r in reasons)


def test_probe_agrees_on_preconditions_that_cannot_be_evaluated(gloo_world, monkeypatch):
    """A rank whose precondition check itself raises (CUDA that cannot initialize, a device
    string torch rejects) must still reach the first agreement: its peers are waiting in it."""

    def broken(group, device):
        raise LookupError("no CUDA context")

    monkeypatch.setattr(symm_mem_pool, "_local_preconditions", broken)
    assert probe_symmetric_memory(gloo_world, "cpu") == (
        False,
        "rank 0: the local preconditions could not be evaluated: LookupError: no CUDA context",
    )
    monkeypatch.undo()
    ok, reason = probe_symmetric_memory(gloo_world, "not-a-device")
    assert ok is False
    assert reason.startswith("rank 0: the local preconditions could not be evaluated: ")


def test_reserve_raises_on_unsupported_group(gloo_world):
    pool = SymmMemPool(gloo_world, torch.device("cpu"), timeout_ms=1000)
    assert pool.rank == 0 and pool.world_size == 1 and pool.timeout_ms == 1000
    assert not pool.reserved
    assert pool.layout is None
    assert (pool.nbytes, pool.slots, pool.slot_nbytes, pool.region_nbytes) == (0, 0, 0, 0)
    assert pool.channels == 0 and pool.offset == 0
    assert "unreserved" in repr(pool)
    with pytest.raises(RuntimeError, match="unsupported group / device.*ProcessGroupNCCL"):
        pool.reserve(4096, slots=2, min_channels=4)
    assert not pool.reserved and pool.generation == 0
    # Nothing works before a reservation, and every message says so.
    for call in (
        lambda: pool.region(0, 0, 0, 1),
        lambda: pool.peer_region(0, 0, 0, 1),
        lambda: pool.signal(0, 0),
        lambda: pool.wait(0, 0),
    ):
        with pytest.raises(RuntimeError, match=r"reserve\(\) first"):
            call()
    pool.release()  # no-op without an allocation
    pool.release()  # idempotent
    assert not pool.reserved


def test_reserve_rejects_bad_requests_before_touching_the_group(gloo_world):
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.timeout_ms == DEFAULT_TIMEOUT_MS and pool.align == DEFAULT_ALIGN
    with pytest.raises(ValueError, match="slot_nbytes >= 1"):
        pool.reserve(0)
    with pytest.raises(ValueError, match="slots >= 1"):
        pool.reserve(1024, slots=0)
    with pytest.raises(ValueError, match="min_channels >= 1"):
        pool.reserve(1024, min_channels=0)


def test_constructor_validation(gloo_world):
    with pytest.raises(ValueError, match="got None"):
        SymmMemPool(None, "cpu")
    with pytest.raises(ValueError, match="timeout_ms >= 0"):
        SymmMemPool(gloo_world, "cpu", timeout_ms=-1)
    assert SymmMemPool(gloo_world, "cpu", timeout_ms=0).timeout_ms == 0  # 0: no deadline
    with pytest.raises(ValueError, match="power of two"):
        SymmMemPool(gloo_world, "cpu", align=100)
    pool = SymmMemPool(gloo_world, "cpu", align=512)
    assert pool.device == torch.device("cpu") and pool.align == 512
    assert pool.group_name == gloo_world.group_name


def test_local_preconditions_name_every_failure(gloo_world, monkeypatch):
    reasons = symm_mem_pool._local_preconditions(gloo_world, torch.device("cpu"))
    assert len(reasons) == 3, reasons
    assert any("needs >= 2 ranks" in r for r in reasons)
    assert any("not a CUDA device" in r for r in reasons)
    assert any("ProcessGroupNCCL" in r for r in reasons)

    def unimportable():
        raise ImportError("no symmetric memory in this build")

    monkeypatch.setattr(symm_mem_pool, "_symm_mem", unimportable)
    reasons = symm_mem_pool._local_preconditions(gloo_world, torch.device("cpu"))
    assert len(reasons) == 4, reasons
    assert "is not importable (no symmetric memory in this build)" in reasons[0]


def test_capture_is_read_from_the_current_cuda_stream(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert symm_mem_pool._is_capturing() is True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert symm_mem_pool._is_capturing() is False


# =============================================================================
# A CPU stand-in for torch's symmetric memory, and ranks as spawned gloo processes
# =============================================================================


class _FakeHandle:
    """What the pool reads from a ``_SymmetricMemory`` handle, over CPU tensors. ``blocks[r]`` is
    rank ``r``'s block as mapped into this process; ``get_buffer`` counts ``storage_offset`` from
    the start of the block, not from the pool tensor, as torch's does."""

    def __init__(self, fake: "_FakeSymmMem", rank: int, blocks: list, pad: torch.Tensor) -> None:
        self._fake, self._blocks, self.pad = fake, blocks, pad
        self.rank, self.world_size = rank, len(blocks)
        self.offset = fake.block_offset
        self.buffer_size = 1 if fake.fail == "handle" else blocks[rank].numel()
        if fake.fail == "short":  # the block holds the layout, but not behind the tensor's offset
            self.buffer_size -= fake.block_offset

    @property
    def signal_pad_size(self) -> int:
        # torch keeps one process-global size: a handle reports it, not its own pad's size.
        self._fake.pad_reads += 1
        return self._fake.pad_nbytes

    def get_buffer(self, rank, sizes, dtype, storage_offset=0):
        if self._fake.fail == "get_buffer":
            raise RuntimeError("injected: get_buffer")
        start = 0 if self._fake.fail == "misaddress" else storage_offset * dtype.itemsize
        view = self._blocks[rank][start : start + sizes[0] * dtype.itemsize].view(dtype)
        self._fake.peer_views.add(view)
        return view

    def get_signal_pad(self, rank, sizes, dtype):
        assert rank == self.rank, "the pool only touches its own rank's pad"
        if sizes[0] * dtype.itemsize > self.pad.numel():
            raise RuntimeError("injected: get_signal_pad past the end of the pad")
        return self.pad[: sizes[0] * dtype.itemsize].view(dtype)

    def put_signal(self, dst_rank, channel=0, timeout_ms=0):
        self._fake.signals.append(("put", dst_rank, channel, timeout_ms))

    def wait_signal(self, src_rank, channel=0, timeout_ms=0):
        self._fake.signals.append(("wait", src_rank, channel, timeout_ms))


class _FakeSymmMem:
    """Stand-in for ``torch.distributed._symmetric_memory`` on CPU.

    ``empty`` creates this rank's block as a file under ``root`` and returns the pool tensor
    ``block_offset`` bytes into it; ``rendezvous`` is a real collective on the group (the ranks
    exchange their file names, as torch's exchanges handles through the group's store) and maps
    every rank's file, so a peer view aliases the peer process's buffer. Every allocation gets
    its own signal pad, handed out dirty. ``fail`` names the step at which this rank fails
    locally, ``error`` what ``empty`` and ``rendezvous`` raise then.
    """

    def __init__(self, root: str, *, block_offset: int = 0, pad_nbytes: int = 9216) -> None:
        self.root, self.block_offset, self.pad_nbytes = str(root), block_offset, pad_nbytes
        self.fail = ""
        self.error: type[Exception] = RuntimeError
        self.pad_reads = 0
        self.signals: list[tuple] = []
        self.handles: weakref.WeakSet = weakref.WeakSet()  # the handles still alive
        self.buffers: weakref.WeakSet = weakref.WeakSet()  # the pool tensors still alive
        self.peer_views: weakref.WeakSet = weakref.WeakSet()  # the get_buffer views still alive
        self.block: torch.Tensor | None = None  # this rank's most recent block and pad
        self.pad: torch.Tensor | None = None
        self._allocations: dict[int, tuple[str, torch.Tensor]] = {}
        self.allocated = 0  # blocks handed out by `empty`

    def empty(self, nbytes, *, dtype, device):
        if self.fail == "empty":
            raise self.error("injected: out of memory")
        self.allocated += 1
        path = os.path.join(self.root, f"block_{os.getpid()}_{self.allocated}")
        self.block = torch.from_file(
            path, shared=True, size=self.block_offset + nbytes, dtype=dtype
        )
        self.pad = torch.full((self.pad_nbytes,), 255, dtype=torch.uint8)
        buf = self.block[self.block_offset :]
        self._allocations[buf.data_ptr()] = (path, self.pad)
        self.buffers.add(buf)
        return buf

    def rendezvous(self, buf, group):
        path, pad = self._allocations[buf.data_ptr()]
        paths = [None] * dist.get_world_size(group)
        dist.all_gather_object(paths, path, group=group)
        if self.fail == "rendezvous":
            raise self.error("injected: rendezvous")
        blocks = [
            torch.from_file(p, shared=True, size=os.path.getsize(p), dtype=torch.uint8)
            for p in paths
        ]
        handle = _FakeHandle(self, dist.get_rank(group), blocks, pad)
        if self.fail == "rank":
            handle.rank += 1
        if self.fail == "world":
            handle.world_size += 1
        self.handles.add(handle)
        return handle

    def get_backend(self, device):
        return "FAKE"


def _use_fake(patch, root, **kwargs) -> _FakeSymmMem:
    """Run the pool on the CPU fake: its symmetric-memory module is replaced and what needs a
    CUDA device or an NCCL group (the local preconditions, the capture query, the device
    synchronize) is stubbed. ``patch`` is ``monkeypatch.setattr`` in a test and ``setattr`` in a
    spawned rank."""
    fake = _FakeSymmMem(root, **kwargs)
    patch(symm_mem_pool, "_symm_mem", lambda: fake)
    patch(symm_mem_pool, "_local_preconditions", lambda group, device: [])
    patch(symm_mem_pool, "_is_capturing", lambda: False)
    patch(torch.cuda, "synchronize", lambda device=None: None)
    return fake


@pytest.fixture
def fake_symm(gloo_world, monkeypatch, tmp_path):
    return _use_fake(monkeypatch.setattr, tmp_path)


@pytest.fixture
def collectives(monkeypatch):
    """The names of the group collectives issued from here on, in order."""
    calls: list[str] = []
    for name in ("all_gather_object", "barrier"):

        def recording(*args, _name=name, _real=getattr(dist, name), **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(dist, name, recording)
    return calls


def _rank_main(rank: int, init_file: str, root: str, scenario, *args) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),  # a rank left alone in a collective fails, not hangs
    )
    try:
        scenario(rank, dist.group.WORLD, root, *args)
    finally:
        dist.destroy_process_group()


def _spawn_ranks(tmp_path, scenario, *args) -> None:
    """Run ``scenario(rank, group, root, *args)`` on two spawned gloo ranks (they rendezvous
    through a file, no port) and re-raise a rank's exception; bounded, so a hang fails the
    test."""
    ctx = mp.spawn(
        _rank_main,
        args=(str(tmp_path / "init"), str(tmp_path), scenario, *args),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 180
    try:
        # ProcessContext.join(timeout) returns as soon as one process exits: loop until all have.
        while not ctx.join(timeout=5):
            assert time.monotonic() < deadline, f"{scenario.__name__} did not finish within 180 s"
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.kill()


# =============================================================================
# Two gloo ranks that cannot carry the pool: the probe declines on both, with both reasons
# =============================================================================


def _probe_declines_on_both_ranks(rank: int, group, root: str) -> None:
    # The probe declines on every rank, and every rank's reason names every rank.
    ok, reason = probe_symmetric_memory(group, torch.device("cpu"))
    assert ok is False
    assert "rank 0: " in reason and "rank 1: " in reason, reason
    assert "ProcessGroupNCCL" in reason and "needs >= 2 ranks" not in reason
    # MIN semantics: one dissenting rank fails everyone, and the reason names it.
    ok, reason = symm_mem_pool._agree(group, rank != 1, "" if rank != 1 else "dissent")
    assert (ok, reason) == (False, "rank 1: dissent")
    assert symm_mem_pool._agree(group, True, "") == (True, "")
    # A pool takes its rank and size from its own group, not from the default one. Creating a
    # group is collective on the default group; only rank 1 is a member of this one.
    solo = dist.new_group([1])
    if rank == 1:
        pool = SymmMemPool(solo, "cpu")
        assert (pool.rank, pool.world_size, pool.group_name) == (0, 1, solo.group_name)


def test_probe_and_agreement_two_gloo_ranks(tmp_path):
    _spawn_ranks(tmp_path, _probe_declines_on_both_ranks)


# =============================================================================
# Two ranks share a pool: agreed layout, peer views, signals, growth
# =============================================================================


def _two_ranks_share_a_pool(rank: int, group, root: str) -> None:
    fake = _use_fake(setattr, root, block_offset=512)
    peer = 1 - rank
    pool = SymmMemPool(group, "cpu", timeout_ms=1234)

    # The ranks ask for different sizes and both get the larger of each:
    #   rank 0: 2 slots of 1000 bytes -> 500 bytes per rank, regions of 512
    #   rank 1: 3 slots of 3000 bytes -> 1500 bytes per rank, regions of 1536
    assert pool.reserve(*[(1000, 2), (3000, 3)][rank]) is True
    assert pool.layout == PoolLayout(world_size=2, slots=3, region_nbytes=1536)
    assert (pool.slots, pool.region_nbytes, pool.slot_nbytes, pool.nbytes) == (3, 1536, 3072, 9216)
    assert pool.reserved and pool.generation == 1 and pool.offset == 512
    assert pool.channels == 1152  # 9216 pad bytes / 4 per flag / 2 ranks
    assert fake.pad.numel() == 9216 and not fake.pad.any()  # handed out dirty, cleared here

    # Producer: this rank's region of slot 1 in the PEER's buffer (and of slot 2 in its own).
    mine = [10 * (rank + 1) + i for i in range(8)]
    pool.peer_region(peer, 1, 16, 8).copy_(torch.tensor(mine, dtype=torch.uint8))
    pool.peer_region(rank, 2, 0, 4).fill_(7 + rank)
    dist.barrier(group=group)
    # Consumer: the peer's region of slot 1 in THIS rank's buffer holds what the peer pushed.
    theirs = [10 * (peer + 1) + i for i in range(8)]
    assert pool.region(1, peer, 16, 8).tolist() == theirs
    assert pool.region(2, rank, 0, 4).tolist() == [7 + rank] * 4
    # The same bytes by their position in this rank's block file. The pool tensor starts 512
    # bytes in, slot 1 at 3072, slot 2 at 6144; rank 1's region is 1536 bytes into a slot.
    #   rank 0 holds rank 1's push at 512 + 3072 + 1536 + 16, its own at 512 + 6144
    #   rank 1 holds rank 0's push at 512 + 3072 + 16, its own at 512 + 6144 + 1536
    pushed, own = [(5136, 6656), (3600, 8192)][rank]
    assert fake.block[pushed : pushed + 8].tolist() == theirs
    assert fake.block[own : own + 4].tolist() == [7 + rank] * 4
    assert int(fake.block.count_nonzero()) == 12  # nothing else: the check's markers are gone

    # Signals carry the pool's deadline; ranks and channels are bounds-checked first.
    pool.signal(peer, 3)
    pool.wait(peer, 1151)
    assert fake.signals == [("put", peer, 3, 1234), ("wait", peer, 1151, 1234)]
    for bad_rank in (
        lambda: pool.signal(2, 0),
        lambda: pool.wait(-1, 0),
        lambda: pool.peer_region(2, 0, 0, 1),
    ):
        with pytest.raises(ValueError, match="out of range for 2"):
            bad_rank()
    for bad_channel in (lambda: pool.signal(peer, 1152), lambda: pool.wait(peer, -1)):
        with pytest.raises(ValueError, match=r"out of range \(the pool offers 1152 channels\)"):
            bad_channel()
    assert len(fake.signals) == 2

    # A request the allocation covers (again different on each rank) changes nothing.
    assert pool.reserve(*[(3072, 3), (16, 1)][rank]) is False
    assert pool.generation == 1 and pool.nbytes == 9216 and fake.allocated == 1
    # The channel need is the largest any rank asked for as well: rank 1's 2000 exceed the 1152
    # of both pads, so both ranks report a shortfall, and the allocation stays.
    with pytest.raises(RuntimeError) as info:
        pool.reserve(16, 1, min_channels=(1, 2000)[rank])
    assert "rank 0: the signal pad offers 1152 channels for 2 ranks; need 2000" in str(info.value)
    assert "rank 1: the signal pad offers 1152 channels" in str(info.value)
    assert pool.reserved and pool.generation == 1

    # The ranks must hold the pool in the same state: what a reserve does next, and the
    # collectives it issues, depend on it. Rank 1 comes with a new pool object while rank 0
    # holds the allocation: both raise, and both keep what they had.
    stray = pool if rank == 0 else SymmMemPool(group, "cpu")
    with pytest.raises(
        RuntimeError, match=r"ranks disagree on the pool's state .*, 1\), \(None, 0\)"
    ):
        stray.reserve(16, 1)
    assert pool.reserved and pool.generation == 1 and stray.reserved == (rank == 0)

    # Rank 0 now needs 4096-byte regions, in a single slot: the pool grows to the union of the
    # request and what it has, 3 slots of 4096-byte regions, in a new block with a cleared pad.
    assert pool.reserve(*[(8192, 1), (100, 1)][rank]) is True
    assert pool.layout == PoolLayout(world_size=2, slots=3, region_nbytes=4096)
    assert pool.nbytes == 24576 and pool.generation == 2 and fake.allocated == 2
    assert not fake.pad.any() and not fake.block.any()

    # Pools that disagree on the alignment cannot share a layout: both ranks raise.
    other = SymmMemPool(group, "cpu", align=(256, 512)[rank])
    with pytest.raises(ValueError, match="ranks disagree on align"):
        other.reserve(4096)
    assert not other.reserved

    pool.release()
    assert not pool.reserved and pool.layout is None
    assert len(fake.handles) == 0 and len(fake.buffers) == 0
    assert (pool.nbytes, pool.slots, pool.slot_nbytes, pool.region_nbytes) == (0, 0, 0, 0)
    assert pool.channels == 0 and pool.offset == 0 and pool.generation == 2
    with pytest.raises(RuntimeError, match=r"reserve\(\) first"):
        pool.region(0, 0, 0, 1)
    # Unreserved on both ranks, but not the same pool: rank 1's new object has no history.
    stray = pool if rank == 0 else SymmMemPool(group, "cpu")
    with pytest.raises(RuntimeError, match=r"the pool's state .*\(None, 2\), \(None, 0\)"):
        stray.reserve(16, 1)
    # Nor is an allocation that one rank dropped on its own: the same count, one of them gone.
    lone = SymmMemPool(group, "cpu")
    assert lone.reserve(16, 1) is True
    if rank == 1:
        lone._free(fence=False)
    with pytest.raises(RuntimeError, match=r"the pool's state .*, 1\), \(None, 1\)"):
        lone.reserve(16, 1)
    if rank == 0:
        lone.release()
    else:
        dist.barrier(group=group)  # the fence of rank 0's release

    _first_push_meets_a_slow_peer(rank, group)


def _first_push_meets_a_slow_peer(rank: int, group) -> None:
    """Rank 0 pushes into rank 1's slot 0 as soon as its ``reserve`` returns, while rank 1 has
    not yet come back from the allocation's last collective. The push must arrive whole: rank 1
    has no clearing of the addressing check's markers left to do then."""
    pool = SymmMemPool(group, "cpu")
    payload = [0xAB] * 16
    if rank == 0:
        assert pool.reserve(4096, slots=1) is True
        pool.peer_region(1, 0, 0, 16).copy_(torch.tensor(payload, dtype=torch.uint8))
        dist.barrier(group=group)  # lets rank 1 go on, now that the push is in its buffer
    else:
        real_barrier, seen = dist.barrier, []

        def held_back_after_the_last_fence(*args, **kwargs):
            real_barrier(*args, **kwargs)
            seen.append(1)
            if len(seen) == 2:  # the fences of an allocation: after the push, after the read
                real_barrier(*args, **kwargs)  # stay here until rank 0 has pushed

        dist.barrier = held_back_after_the_last_fence
        try:
            assert pool.reserve(4096, slots=1) is True
        finally:
            dist.barrier = real_barrier
        assert len(seen) == 2
        assert pool.region(0, 0, 0, 16).tolist() == payload
    pool.release()


def test_two_ranks_share_a_pool(tmp_path):
    """Two ranks with different requests end with one layout; what a rank writes through
    ``peer_region`` is what its peer reads through ``region``, with the pool tensor at a
    non-zero offset inside its block; a covered request is a no-op and a larger one regrows;
    pools in different states refuse to reserve; a push right after ``reserve`` is not lost."""
    _spawn_ranks(tmp_path, _two_ranks_share_a_pool)


# =============================================================================
# A local failure on one rank is agreed before any rank raises
# =============================================================================

_REGION = SymmMemPool.region


def _injected_region(self, *args, **kwargs):
    raise RuntimeError("injected: region")


# What rank 1 fails at -> the reason EVERY rank must end with. "misaddress" is rank 1 writing its
# markers through a wrong peer view, which rank 0 is the one to notice.
_LOCAL_FAILURES = {
    "empty": "rank 1: symm_mem.empty failed: RuntimeError: injected: out of memory",
    "rendezvous": "rank 1: rendezvous failed: RuntimeError: injected: rendezvous",
    "handle": "rank 1: the symmetric block holds 1 bytes",
    "pad": "rank 1: the signal pad offers 1 channels for 2 ranks; need 2",
    "get_buffer": "rank 1: marker push failed: RuntimeError: injected: get_buffer",
    "read": "rank 1: marker read failed: RuntimeError: injected: region",
    "misaddress": "rank 0: read markers",
}


def _inject(fake: _FakeSymmMem, step: str) -> None:
    if step == "pad":
        fake.pad_nbytes = 8  # 8 bytes / 4 per flag / 2 ranks = 1 channel
    elif step == "read":
        SymmMemPool.region = _injected_region
    else:
        fake.fail = step


def _lift(fake: _FakeSymmMem) -> None:
    fake.fail, fake.pad_nbytes, SymmMemPool.region = "", 9216, _REGION


def _one_rank_fails_locally(rank: int, group, root: str) -> None:
    fake = _use_fake(setattr, root, block_offset=512)
    for via in ("reserve", "probe"):
        for step, expected in _LOCAL_FAILURES.items():
            where = (via, step, rank)
            if rank == 1:
                _inject(fake, step)
            pool = SymmMemPool(group, "cpu")
            if via == "reserve":
                with pytest.raises(RuntimeError) as info:
                    pool.reserve(4096, slots=2, min_channels=2)
                reason = str(info.value)
                assert not pool.reserved and pool.layout is None and pool.nbytes == 0, where
            else:
                ok, reason = probe_symmetric_memory(group, "cpu", min_channels=2)
                assert ok is False, where
            unnamed = 1 if expected.startswith("rank 0") else 0
            assert expected in reason and f"rank {unnamed}:" not in reason, (where, reason)
            # Nothing is left mapped on any rank, the peer views the pool cached included ...
            assert len(fake.handles) == 0 and len(fake.buffers) == 0, where
            assert len(fake.peer_views) == 0, where
            # ... every rank holds the same verdict and the group's next collective still pairs ...
            verdicts = [None, None]
            dist.all_gather_object(verdicts, reason, group=group)
            assert verdicts[0] == verdicts[1], (where, verdicts)
            # ... and with the fault gone a pool reserves on every rank.
            _lift(fake)
            assert pool.reserve(4096, slots=2, min_channels=2) is True, where
            assert pool.reserved and (pool.slots, pool.region_nbytes) == (2, 2048), where
            pool.release()
            assert not pool.reserved and len(fake.handles) == 0, where
            assert len(fake.peer_views) == 0, where

    # A request the allocation covers, needing more channels than the pad has (the same small
    # pad on both ranks): agreed as well, naming both ranks, and the allocation stays.
    _inject(fake, "pad")
    pool = SymmMemPool(group, "cpu")
    assert pool.reserve(4096, slots=2) is True
    with pytest.raises(RuntimeError) as info:
        pool.reserve(4096, slots=2, min_channels=2)
    shortfall = "the signal pad offers 1 channels for 2 ranks; need 2"
    assert f"rank 0: {shortfall}" in str(info.value) and f"rank 1: {shortfall}" in str(info.value)
    assert pool.reserved and pool.generation == 1
    _lift(fake)

    # Pad sizes that differ inside one of torch's 128-byte steps (its rendezvous compares the
    # rounded size, so it lets these through): the ranks would count 1152 and 1150 channels, so
    # the allocation fails with the sizes by rank, through reserve and through the probe.
    if rank == 1:
        fake.pad_nbytes = 9200
    differ = "the ranks differ on the signal pad size in bytes: [9216, 9200] by rank"
    other = SymmMemPool(group, "cpu")
    with pytest.raises(RuntimeError, match=re.escape(f"SymmMemPool.reserve: {differ}")):
        other.reserve(4096, slots=2, min_channels=2)
    assert not other.reserved and other.generation == 0
    assert probe_symmetric_memory(group, "cpu", min_channels=2) == (False, differ)
    # Only the pool reserved above is still mapped.
    assert len(fake.handles) == 1 and len(fake.buffers) == 1
    _lift(fake)

    # A pool that has to grow, and rank 1 cannot allocate the larger block: the old allocation
    # is gone on both ranks, the count of agreed allocations stands, and a retry is a fresh one.
    if rank == 1:
        _inject(fake, "empty")
    with pytest.raises(RuntimeError, match=re.escape(_LOCAL_FAILURES["empty"])):
        pool.reserve(8192, slots=2)
    assert not pool.reserved and pool.generation == 1
    assert len(fake.handles) == 0 and len(fake.buffers) == 0
    _lift(fake)
    assert pool.reserve(8192, slots=2) is True
    assert pool.generation == 2 and (pool.slots, pool.region_nbytes) == (2, 4096)
    pool.release()


def test_one_rank_fails_locally_every_rank_gets_the_agreed_reason(tmp_path):
    """Rank 1 fails at each local step of an allocation in turn (``symm_mem.empty``, the
    rendezvous, a handle check, a too-small pad, the marker push, the marker read, a wrong peer
    view), through ``reserve`` and through the probe, and then while its pool grows. Both ranks
    must end with the same reason and nothing reserved, without waiting in a collective the
    other rank never enters, and must be able to reserve afterwards."""
    _spawn_ranks(tmp_path, _one_rank_fails_locally)


@pytest.mark.parametrize("via", ["reserve", "probe"])
def test_a_failed_collective_drops_the_allocation_and_propagates(
    gloo_world, fake_symm, monkeypatch, via
):
    """When the group itself fails after the rendezvous (here: its barrier), nothing can be
    agreed: the rank drops its allocation without another collective and the error propagates."""
    calls: list[str] = []

    def broken_barrier(group=None):
        calls.append("barrier")
        raise RuntimeError("injected: the group's barrier failed")

    monkeypatch.setattr(dist, "barrier", broken_barrier)
    pool = SymmMemPool(gloo_world, "cpu")
    with pytest.raises(RuntimeError, match="the group's barrier failed") as info:
        if via == "reserve":
            pool.reserve(4096)
        else:
            probe_symmetric_memory(gloo_world, "cpu")
    # The frames of the traceback are still alive here, and none of them holds the block.
    assert "_allocate" in [entry.name for entry in info.traceback]
    assert len(fake_symm.handles) == 0 and len(fake_symm.buffers) == 0
    assert calls == ["barrier"]
    if via == "reserve":
        assert not pool.reserved


def test_probe_passes_on_a_healthy_group_and_leaves_nothing_mapped(
    gloo_world, fake_symm, collectives
):
    assert probe_symmetric_memory(gloo_world, "cpu", min_channels=4) == (True, "")
    # One region of 256 bytes for the one rank, and nothing of it is left. The probe's pool is
    # a local, so its block would go with it anyway: what shows the release is its fence, the
    # third after the two of the addressing check.
    assert fake_symm.allocated == 1 and fake_symm.block.numel() == 256
    assert len(fake_symm.handles) == 0 and len(fake_symm.buffers) == 0
    assert collectives.count("barrier") == 3
    # One rank and a 9216-byte pad of 4-byte flags: 2304 channels, one too few.
    ok, reason = probe_symmetric_memory(gloo_world, "cpu", min_channels=2305)
    assert ok is False
    assert reason.startswith("rank 0: the signal pad offers 2304 channels for 1 ranks; need 2305")
    assert len(fake_symm.handles) == 0


_GATHER, _BARRIER = "all_gather_object", "barrier"


def test_an_allocation_issues_the_same_collectives_whatever_fails(
    gloo_world, fake_symm, collectives
):
    """The collectives of one allocation, in order, for a success and for a failure at each
    stage. A rank cannot tell which of these its peers are in, so each row must be a prefix of
    the success row, plus at most the fence of the free."""
    # reserve: request gather | agreement after empty | the stand-in's rendezvous exchange |
    # agreement after the rendezvous | marker push agreed | fence | marker read agreed | fence
    success = [_GATHER] * 5 + [_BARRIER, _GATHER, _BARRIER]

    def reserve_failing_at(step: str) -> list[str]:
        del collectives[:]
        _inject(fake_symm, step)
        try:
            pool = SymmMemPool(gloo_world, "cpu")
            if step:
                with pytest.raises(RuntimeError, match="SymmMemPool.reserve: .*rank 0: "):
                    pool.reserve(4096)
                assert not pool.reserved
            else:
                assert pool.reserve(4096) is True
        finally:
            _lift(fake_symm)
        return list(collectives)

    assert reserve_failing_at("") == success
    assert reserve_failing_at("empty") == success[:2]  # nobody enters the rendezvous
    assert reserve_failing_at("rendezvous") == success[:4]
    assert reserve_failing_at("handle") == success[:4]
    # A failed read: the peers pushed markers into this buffer, so it is freed behind a fence.
    assert reserve_failing_at("read") == success + [_BARRIER]
    # The ranks had adopted that allocation, so it counted; one that fails earlier does not.
    counted = SymmMemPool(gloo_world, "cpu")
    for step, generation in (("read", 1), ("handle", 1), ("empty", 1)):
        _inject(fake_symm, step)
        with pytest.raises(RuntimeError, match="SymmMemPool.reserve: .*rank 0: "):
            counted.reserve(4096)
        _lift(fake_symm)
        assert (counted.generation, counted.reserved) == (generation, False)
    # Growth: the request gather, the fence that frees the old allocation, the fence behind
    # which every rank has dropped it, a new allocation.
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.reserve(4096) is True
    del collectives[:]
    assert pool.reserve(8192) is True
    assert collectives == [_GATHER, _BARRIER, _BARRIER] + success[1:]
    # A growth whose new allocation fails stops where a first allocation would.
    fake_symm.fail = "empty"
    del collectives[:]
    with pytest.raises(RuntimeError, match="rank 0: symm_mem.empty failed"):
        pool.reserve(16384)
    assert collectives == [_GATHER, _BARRIER, _BARRIER, _GATHER] and not pool.reserved
    fake_symm.fail = ""
    assert pool.reserve(8192) is True
    # A covered request: the request gather and the agreement on the channels.
    del collectives[:]
    assert pool.reserve(4096) is False
    assert collectives == [_GATHER, _GATHER]
    # The probe: its precondition agreement in place of the request gather, and the fence of
    # the release at the end.
    del collectives[:]
    assert probe_symmetric_memory(gloo_world, "cpu") == (True, "")
    assert collectives == success + [_BARRIER]


def test_each_device_phase_is_drained_before_its_verdict_is_agreed(
    gloo_world, fake_symm, collectives, monkeypatch
):
    """On a device the pad clearing and the marker writes are asynchronous. The pool
    synchronizes after each, so that an error surfaces in the phase whose verdict is agreed next
    and nothing is in flight when a rank drops its block; the markers are cleared and drained
    before the last fence, behind which a peer may push; a free drains before it fences."""
    sync, pad, push, local = "synchronize", "get_signal_pad", "peer_region", "region"
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: collectives.append(sync))
    for owner, name in ((_FakeHandle, pad), (SymmMemPool, push), (SymmMemPool, local)):

        def recording(*args, _name=name, _real=getattr(owner, name), **kwargs):
            collectives.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(owner, name, recording)
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.reserve(4096) is True
    assert collectives == [
        *[_GATHER, _GATHER, _GATHER],  # requests | empty agreed | the rendezvous exchange
        *[pad, sync, _GATHER],  # pad cleared, then drained | rendezvous agreed
        *[push, sync, _GATHER, _BARRIER],  # markers pushed, then drained | push agreed | fence
        *[local, _GATHER],  # markers read | read agreed
        *[local, sync, _BARRIER],  # markers cleared, then drained | fence
    ]
    del collectives[:]
    pool.release()
    assert collectives == [sync, _BARRIER]


_HANDLE_DEFECTS = {
    "rank": "rank 0: rendezvous returned rank 1/1, expected 0/1",
    "world": "rank 0: rendezvous returned rank 0/2, expected 0/1",
    "handle": "rank 0: the symmetric block holds 1 bytes, the pool needs 4096 at offset 0",
    "short": "rank 0: the symmetric block holds 4096 bytes, the pool needs 4096 at offset 512",
    "offset": "rank 0: the pool tensor sits at byte offset 100 of its symmetric block",
}


@pytest.mark.parametrize("defect", _HANDLE_DEFECTS)
def test_reserve_rejects_a_handle_that_cannot_carry_the_layout(gloo_world, fake_symm, defect):
    if defect == "offset":
        fake_symm.block_offset = 100  # not a multiple of the 256-byte alignment
    else:
        fake_symm.fail = defect
        fake_symm.block_offset = 512 if defect == "short" else 0
    pool = SymmMemPool(gloo_world, "cpu")
    with pytest.raises(RuntimeError, match=re.escape(_HANDLE_DEFECTS[defect])):
        pool.reserve(4096, slots=1)
    assert not pool.reserved and pool.generation == 0
    assert len(fake_symm.handles) == 0 and len(fake_symm.buffers) == 0


@pytest.mark.parametrize("step", ["empty", "rendezvous", "read"])
def test_whatever_a_local_step_raises_becomes_the_agreed_reason(
    gloo_world, fake_symm, monkeypatch, step
):
    """Not only ``RuntimeError``: a rank that let any other exception of a local step escape
    would raise alone and leave its peers in the next collective."""
    fake_symm.error = LookupError
    if step == "read":

        def failing_region(self, *args, **kwargs):
            raise LookupError("injected: region")

        monkeypatch.setattr(SymmMemPool, "region", failing_region)
    else:
        fake_symm.fail = step
    pool = SymmMemPool(gloo_world, "cpu")
    with pytest.raises(RuntimeError, match="rank 0: .* failed: LookupError: injected: "):
        pool.reserve(4096)
    assert not pool.reserved and len(fake_symm.handles) == 0 and len(fake_symm.buffers) == 0


def test_every_collective_runs_on_the_pools_own_group(gloo_world, fake_symm, monkeypatch):
    """A pool on a sub-group must not touch the default group: every gather, every fence and
    the rendezvous run on the group the pool was built with."""
    sub_group = dist.new_group([0], backend="gloo")
    groups: list = []
    for name in ("all_gather_object", "barrier"):

        def recording(*args, _real=getattr(dist, name), **kwargs):
            groups.append(kwargs.get("group"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(dist, name, recording)
    pool = SymmMemPool(sub_group, "cpu")
    assert pool.reserve(4096) is True  # an allocation: gathers, the rendezvous, the fences
    assert pool.reserve(8192) is True  # growth: the fences around the free
    assert pool.reserve(16) is False  # covered: the agreement on the channels
    pool.release()
    assert probe_symmetric_memory(sub_group, "cpu") == (True, "")
    assert len(groups) > 20 and all(group is sub_group for group in groups)


def test_reserve_carves_with_the_pools_alignment(gloo_world, fake_symm):
    # 700 bytes for the one rank: 1024 in 512-byte steps (768 with the default 256).
    pool = SymmMemPool(gloo_world, "cpu", align=512)
    assert pool.reserve(700, slots=1) is True
    assert pool.layout == PoolLayout(world_size=1, slots=1, region_nbytes=1024, align=512)


def test_the_stand_in_follows_torchs_symmetric_memory_api():
    """The stand-in above is written by hand, and the allocation behind it has no test without
    a device: tie the names and the argument order the pool relies on to the torch under test."""
    import torch.distributed._symmetric_memory as symm_mem

    assert list(inspect.signature(symm_mem.empty).parameters) == ["size", "dtype", "device"]
    assert list(inspect.signature(symm_mem.rendezvous).parameters) == ["tensor", "group"]
    assert list(inspect.signature(symm_mem.get_backend).parameters) == ["device"]
    handle = torch._C._distributed_c10d._SymmetricMemory
    for attribute in ("rank", "world_size", "offset", "buffer_size", "signal_pad_size"):
        assert isinstance(inspect.getattr_static(handle, attribute), property), attribute
    positional = {
        "get_buffer": ["rank", "sizes", "dtype", "storage_offset"],
        "get_signal_pad": ["rank", "sizes", "dtype"],
        "put_signal": ["dst_rank", "channel", "timeout_ms"],
        "wait_signal": ["src_rank", "channel", "timeout_ms"],
    }
    for method, expected in positional.items():
        signature = getattr(handle, method).__doc__.splitlines()[0]  # written by pybind11
        names = re.findall(r"(\w+): ", signature)
        assert names[: len(expected) + 1] == ["self", *expected], signature


# =============================================================================
# Signals and the signal pad
# =============================================================================


def test_allocation_clears_the_whole_signal_pad(gloo_world, fake_symm):
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.reserve(4096) is True
    # The stand-in hands every pad out with all 9216 bytes set.
    assert fake_symm.pad.numel() == 9216 and not fake_symm.pad.any()


def test_channel_count_is_fixed_when_the_pool_allocates(gloo_world, fake_symm):
    """torch keeps one process-wide signal pad size and a handle reports that, not the size its
    own pad was allocated with. The pool reads it once per allocation, so a later
    ``set_signal_pad_size`` cannot admit channels that lie past this allocation's pad."""
    pool = SymmMemPool(gloo_world, "cpu", timeout_ms=77)
    assert pool.channels == 0
    assert pool.reserve(1000) is True
    assert fake_symm.pad_reads == 1
    assert pool.channels == 2304  # 9216 bytes / 4 per flag / 1 rank
    fake_symm.pad_nbytes = 4 * 9216  # the process-wide size grows; this allocation's pad did not
    assert pool.channels == 2304
    pool.signal(0, 2303)
    pool.wait(0, 2303)
    assert fake_symm.signals == [("put", 0, 2303, 77), ("wait", 0, 2303, 77)]
    with pytest.raises(ValueError, match="channel 2304 out of range"):
        pool.signal(0, 2304)
    with pytest.raises(ValueError, match="channel 2304 out of range"):
        pool.wait(0, 2304)
    # A request the allocation covers is checked against the allocation's channels as well.
    with pytest.raises(RuntimeError, match="rank 0: the signal pad offers 2304 channels .* 2305"):
        pool.reserve(1000, min_channels=2305)
    assert pool.reserved and fake_symm.pad_reads == 1
    # The next allocation has a pad of the size current at that time.
    assert pool.reserve(4000) is True
    assert fake_symm.pad_reads == 2 and pool.channels == 9216
    pool.release()
    assert pool.channels == 0


# =============================================================================
# CUDA-graph capture
# =============================================================================


def test_reserve_release_and_probe_refuse_to_run_under_capture(
    gloo_world, fake_symm, collectives, monkeypatch
):
    """All three synchronize the device and run group collectives, which a CUDA graph cannot
    capture, and two of them unmap memory a graph may hold: each raises before doing any of
    it."""
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.reserve(4096) is True
    del collectives[:]
    monkeypatch.setattr(symm_mem_pool, "_is_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="SymmMemPool.reserve under CUDA-graph capture"):
        pool.reserve(8192)
    with pytest.raises(RuntimeError, match="SymmMemPool.release under CUDA-graph capture"):
        pool.release()
    with pytest.raises(RuntimeError, match="probe_symmetric_memory under CUDA-graph capture"):
        probe_symmetric_memory(gloo_world, "cpu")
    assert collectives == [] and fake_symm.allocated == 1
    assert pool.reserved and pool.generation == 1 and len(fake_symm.handles) == 1
    SymmMemPool(gloo_world, "cpu").release()  # nothing reserved: a no-op, also under capture
    monkeypatch.setattr(symm_mem_pool, "_is_capturing", lambda: False)
    pool.release()
    assert not pool.reserved and collectives == ["barrier"]


# =============================================================================
# Release
# =============================================================================


@pytest.mark.parametrize("error", [RuntimeError, AssertionError])
def test_release_fences_and_clears_when_the_device_synchronize_fails(
    gloo_world, fake_symm, collectives, monkeypatch, error
):
    """A device error on this rank, of whatever type, must not keep it out of the fence its
    peers are waiting in, nor leave the pool half released."""
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.reserve(4096) is True
    del collectives[:]

    def failing_synchronize(device=None):
        raise error("injected: CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(torch.cuda, "synchronize", failing_synchronize)
    pool.release()
    assert collectives == ["barrier"]
    assert not pool.reserved and pool.layout is None
    assert len(fake_symm.handles) == 0 and len(fake_symm.buffers) == 0
    assert (pool.nbytes, pool.channels, pool.offset) == (0, 0, 0)
    pool.release()  # nothing left to free: no second fence
    assert collectives == ["barrier"]


def test_release_clears_even_when_the_fence_itself_fails(gloo_world, fake_symm, monkeypatch):
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.reserve(4096) is True

    def broken_barrier(group=None):
        raise RuntimeError("injected: the group's barrier failed")

    monkeypatch.setattr(dist, "barrier", broken_barrier)
    with pytest.raises(RuntimeError, match="the group's barrier failed") as info:
        pool.release()
    # The frames of the traceback are still alive here, and none of them holds the block.
    assert "_free" in [entry.name for entry in info.traceback]
    assert not pool.reserved
    assert len(fake_symm.handles) == 0 and len(fake_symm.buffers) == 0
    pool.release()  # nothing left: no second attempt at the fence


def test_release_after_the_group_was_destroyed_clears_without_a_collective(
    gloo_world, fake_symm, collectives
):
    group = dist.new_group([0], backend="gloo")
    pool = SymmMemPool(group, "cpu")
    assert pool.reserve(4096) is True
    dist.destroy_process_group(group)
    del collectives[:]
    pool.release()  # no fence is possible on a group that no longer exists
    assert collectives == []
    assert not pool.reserved and pool.layout is None
    assert len(fake_symm.handles) == 0 and len(fake_symm.buffers) == 0
    pool.release()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
