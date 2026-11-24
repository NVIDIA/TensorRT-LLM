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
"""CPU tests for the symmetric-memory pool (no GPU): the layout arithmetic against a pure-Python
reference, the rank-agreed probe declining a gloo group without CUDA, ``reserve`` raising on an
unsupported group, and the pool's input checks (the two-rank case rendezvous through a file, no
port). The GPU protocol (rendezvous, peer views, signals) is covered by the token-sharded TP
multi-GPU tests that use the pool.
"""

import contextlib
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.distributed import symm_mem_pool
from tensorrt_llm._torch.distributed.symm_mem_pool import (
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
    with pytest.raises(ValueError, match="power of two"):
        PoolLayout(4, 1, 512, align=100)
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
    with pytest.raises(ValueError, match="different pools"):
        a.merged(PoolLayout(2, 2, 1024))
    assert not a.covers(PoolLayout(4, 2, 1024, align=512))


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


def test_reserve_raises_on_unsupported_group(gloo_world):
    pool = SymmMemPool(gloo_world, torch.device("cpu"), timeout_ms=1000)
    assert pool.rank == 0 and pool.world_size == 1 and pool.timeout_ms == 1000
    assert not pool.reserved
    assert pool.layout is None
    assert (pool.nbytes, pool.slots, pool.slot_nbytes, pool.region_nbytes) == (0, 0, 0, 0)
    assert pool.channels == 0 and pool.signal_channels == 0 and pool.offset == 0
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
        pool.barrier,
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
    with pytest.raises(ValueError, match="power of two"):
        SymmMemPool(gloo_world, "cpu", align=100)
    pool = SymmMemPool(gloo_world, "cpu", align=512)
    assert pool.device == torch.device("cpu") and pool.align == 512
    assert pool.group_name == gloo_world.group_name


def test_local_preconditions_name_every_failure(gloo_world):
    reasons = symm_mem_pool._local_preconditions(gloo_world, torch.device("cpu"))
    assert len(reasons) == 3, reasons
    assert any("needs >= 2 ranks" in r for r in reasons)
    assert any("not a CUDA device" in r for r in reasons)
    assert any("ProcessGroupNCCL" in r for r in reasons)


# =============================================================================
# Rank agreement across two gloo ranks (spawned processes, CPU)
# =============================================================================


def _agreement_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        group = dist.group.WORLD
        # The probe declines on every rank, and every rank's reason names every rank.
        ok, reason = probe_symmetric_memory(group, torch.device("cpu"))
        assert ok is False
        assert all(f"rank {r}: " in reason for r in range(world_size)), reason
        assert "ProcessGroupNCCL" in reason and "needs >= 2 ranks" not in reason
        # MIN semantics: one dissenting rank fails everyone, and the reason names it.
        dissenter = world_size - 1
        ok, reason = symm_mem_pool._agree(
            group, rank != dissenter, "" if rank != dissenter else "dissent"
        )
        assert (ok, reason) == (False, f"rank {dissenter}: dissent")
        assert symm_mem_pool._agree(group, True, "") == (True, "")
    finally:
        dist.destroy_process_group()


def test_probe_and_agreement_two_gloo_ranks(tmp_path):
    init_file = str(tmp_path / "rendezvous")
    mp.spawn(_agreement_worker, args=(2, init_file), nprocs=2, join=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Failure paths after an allocation: the pool must not stay reserved (review of #20308)
# =============================================================================


def _fake_allocate(self, layout):
    """Stand-in for ``_allocate`` on a CPU gloo group: records the state a real allocation sets."""
    handle = SimpleNamespace(buffer_size=layout.nbytes, offset=0)
    self._buf, self._handle, self._base_offset, self._layout = object(), handle, 0, layout
    self._peer_buffers.clear()
    self.generation += 1


@pytest.fixture
def cpu_allocating_pool(gloo_world, monkeypatch):
    """A pool whose preconditions, allocation and device syncs are stubbed so the post-allocation
    failure paths of ``reserve`` / ``probe_symmetric_memory`` run on a one-rank gloo group."""
    monkeypatch.setattr(symm_mem_pool, "_local_preconditions", lambda group, device: [])
    monkeypatch.setattr(SymmMemPool, "_allocate", _fake_allocate)
    monkeypatch.setattr(symm_mem_pool.torch.cuda, "synchronize", lambda device=None: None)
    return SymmMemPool(gloo_world, "cpu")


@pytest.mark.parametrize("failing", ["_check_addressing", "_require_channels"])
def test_reserve_frees_the_allocation_when_a_post_allocation_check_fails(
    cpu_allocating_pool, monkeypatch, failing
):
    pool = cpu_allocating_pool
    frees: list[bool] = []
    real_free = SymmMemPool._free

    def recording_free(self, *, fence):
        frees.append(fence)
        real_free(self, fence=fence)

    monkeypatch.setattr(SymmMemPool, "_free", recording_free)
    if failing == "_check_addressing":
        monkeypatch.setattr(SymmMemPool, "_require_channels", lambda self, n: None)
        monkeypatch.setattr(SymmMemPool, "_check_addressing", lambda self: (False, "injected"))
        match = "injected"
    else:

        def no_channels(self, n):
            raise RuntimeError("injected: pad too small")

        monkeypatch.setattr(SymmMemPool, "_require_channels", no_channels)
        match = "pad too small"
    with pytest.raises(RuntimeError, match=match):
        pool.reserve(4096, slots=2, min_channels=4)
    # The failed allocation is gone (fenced free, all ranks reach it together) ...
    assert not pool.reserved and pool.layout is None and frees == [True]
    # ... so a retried request is not mistaken for a covered one: it allocates and fails again
    # instead of returning False on a pool that was never usable.
    with pytest.raises(RuntimeError, match=match):
        pool.reserve(4096, slots=2, min_channels=4)
    assert not pool.reserved and frees == [True, True]


def test_probe_frees_its_allocation_when_the_addressing_check_raises(gloo_world, monkeypatch):
    monkeypatch.setattr(symm_mem_pool, "_local_preconditions", lambda group, device: [])
    monkeypatch.setattr(SymmMemPool, "_allocate", _fake_allocate)
    monkeypatch.setattr(symm_mem_pool.torch.cuda, "synchronize", lambda device=None: None)
    created: list[SymmMemPool] = []
    frees: list[bool] = []
    real_init, real_free = SymmMemPool.__init__, SymmMemPool._free

    def recording_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        created.append(self)

    def recording_free(self, *, fence):
        frees.append(fence)
        real_free(self, fence=fence)

    def boom(self):
        raise RuntimeError("injected CUDA error")

    monkeypatch.setattr(SymmMemPool, "__init__", recording_init)
    monkeypatch.setattr(SymmMemPool, "_free", recording_free)
    monkeypatch.setattr(SymmMemPool, "_check_addressing", boom)
    with pytest.raises(RuntimeError, match="injected CUDA error"):
        probe_symmetric_memory(gloo_world, "cpu")
    assert len(created) == 1 and not created[0].reserved
    assert frees == [False]  # unmapped without a collective: the error may be local to one rank


def test_check_addressing_returns_an_agreed_failure_instead_of_raising(cpu_allocating_pool):
    """A local failure in the marker push (here: the stubbed allocation has no peer buffers) ends the
    check with the agreed reason and no further collective, instead of raising on one rank while
    the others wait at the barrier."""
    pool = cpu_allocating_pool
    pool._allocate(PoolLayout(pool.world_size, 1, pool.align, pool.align))
    ok, reason = pool._check_addressing()
    assert ok is False and reason.startswith("rank 0: marker push failed: ")


def test_reserve_turns_a_local_check_failure_into_a_freed_pool(cpu_allocating_pool, monkeypatch):
    pool = cpu_allocating_pool
    frees: list[bool] = []
    real_free = SymmMemPool._free

    def recording_free(self, *, fence):
        frees.append(fence)
        real_free(self, fence=fence)

    monkeypatch.setattr(SymmMemPool, "_free", recording_free)
    monkeypatch.setattr(SymmMemPool, "_channel_shortfall", lambda self, n: "")
    with pytest.raises(RuntimeError, match="peer views do not address.*marker push failed"):
        pool.reserve(4096)
    assert not pool.reserved and frees == [True]


def test_require_channels_is_agreed_before_raising(cpu_allocating_pool, monkeypatch):
    pool = cpu_allocating_pool
    calls: list[tuple[bool, str]] = []
    real_agree = symm_mem_pool._agree

    def recording_agree(group, ok, reason):
        calls.append((bool(ok), reason))
        return real_agree(group, ok, reason)

    monkeypatch.setattr(symm_mem_pool, "_agree", recording_agree)
    monkeypatch.setattr(
        SymmMemPool, "_channel_shortfall", lambda self, n: "injected: 2 channels for 4"
    )
    with pytest.raises(RuntimeError, match="rank 0: injected: 2 channels for 4"):
        pool._require_channels(4)
    assert calls == [
        (False, "injected: 2 channels for 4")
    ]  # the verdict went through the group first


def test_probe_declines_with_the_agreed_reason_and_releases(gloo_world, monkeypatch):
    monkeypatch.setattr(symm_mem_pool, "_local_preconditions", lambda group, device: [])
    monkeypatch.setattr(SymmMemPool, "_allocate", _fake_allocate)
    monkeypatch.setattr(symm_mem_pool.torch.cuda, "synchronize", lambda device=None: None)
    frees: list[bool] = []
    real_free = SymmMemPool._free

    def recording_free(self, *, fence):
        frees.append(fence)
        real_free(self, fence=fence)

    monkeypatch.setattr(SymmMemPool, "_free", recording_free)
    ok, reason = probe_symmetric_memory(gloo_world, "cpu")
    assert ok is False and "marker push failed" in reason
    assert frees == [True]  # the normal path releases with the fence: every rank agreed and is here


# =============================================================================
# Phase alignment across two gloo ranks: one rank fails locally, nobody hangs (review of #20308)
# =============================================================================


def _divergent_worker(rank: int, world_size: int, init_file: str, phase: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),  # a hang fails the test instead of blocking the suite
    )
    try:
        symm_mem_pool._local_preconditions = lambda group, device: []
        SymmMemPool._allocate = _fake_allocate
        symm_mem_pool.torch.cuda.synchronize = lambda device=None: None
        scratch = torch.zeros(64, dtype=torch.uint8)

        def fake_view(self, *args, **kwargs):
            return scratch[: symm_mem_pool._MARKER_BYTES]

        def injected(self, *args, **kwargs):
            raise RuntimeError("injected")

        failing = rank == world_size - 1
        SymmMemPool.peer_region = injected if failing and phase == "push" else fake_view
        SymmMemPool.region = injected if failing and phase == "read" else fake_view
        pool = SymmMemPool(dist.group.WORLD, "cpu")
        pool._allocate(PoolLayout(world_size, 1, pool.align, pool.align))
        ok, reason = pool._check_addressing()  # must return on BOTH ranks, never raise or hang
        assert ok is False, (rank, reason)
        assert f"rank {world_size - 1}: marker {phase} failed: RuntimeError: injected" in reason, (
            reason
        )
        verdicts: list = [None] * world_size
        dist.all_gather_object(verdicts, (ok, reason))
        assert len(set(verdicts)) == 1, verdicts  # the same agreed verdict on every rank
        dist.barrier()  # both ranks arrive here: no rank was left inside the check's collectives
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("phase", ["push", "read"])
def test_check_addressing_one_rank_fails_locally_both_ranks_agree(tmp_path, phase):
    init_file = str(tmp_path / f"rendezvous_{phase}")
    ctx = mp.spawn(_divergent_worker, args=(2, init_file, phase), nprocs=2, join=False)
    # ProcessContext.join(timeout) returns as soon as one process exits; loop until all have
    # (a worker exception is re-raised by join), bounded so a hang fails the test.
    deadline = time.monotonic() + 180
    while not ctx.join(timeout=5):
        assert time.monotonic() < deadline, (
            "the two-rank addressing check did not finish within 180 s"
        )


# =============================================================================
# Pool additions for the copy-engine protocols: live-pad precondition, pad accessor, signal_write
# =============================================================================


class _FakeHandle:
    """What ``_allocate`` and the signal paths read from a ``_SymmetricMemory`` handle, on CPU
    tensors: one uint8 pad per rank, ``signal_pad_ptrs`` their addresses."""

    def __init__(self, rank: int, world_size: int, buf: torch.Tensor, pads: list[torch.Tensor]):
        self.rank, self.world_size = rank, world_size
        self.offset, self.buffer_size = 0, buf.numel()
        self.pads = pads
        self.signal_pad_size = pads[0].numel()
        self.signal_pad_ptrs = [pad.data_ptr() for pad in pads]

    def get_signal_pad(self, rank, sizes=(), dtype=None, storage_offset=0):
        words = self.pads[rank].view(dtype)
        return words[storage_offset : storage_offset + sizes[0]]


class _FakeSymmMem:
    """Stand-in for ``torch.distributed._symmetric_memory``: ``empty`` + ``rendezvous`` on CPU
    tensors, every rendezvous with its own pads or (``shared_pad``) all of them with one set, as
    an allocator that splits a segment would hand out; ``_SymmetricMemory.memset32`` records."""

    def __init__(self, *, pad_nbytes: int = 9216) -> None:
        self.rank, self.world_size, self.pad_nbytes = 0, 1, pad_nbytes
        self.shared_pad = False
        self._shared: list[torch.Tensor] | None = None
        self.handles: list[_FakeHandle] = []
        self.memsets: list[tuple] = []
        fake = self

        class _SymmetricMemory:
            @staticmethod
            def memset32(input, offset, val, count=1):
                fake.memsets.append((input.data_ptr(), input.dtype, offset, val, count))

        self._SymmetricMemory = _SymmetricMemory

    def _pads(self) -> list[torch.Tensor]:
        fresh = [torch.zeros(self.pad_nbytes, dtype=torch.uint8) for _ in range(self.world_size)]
        if not self.shared_pad:
            return fresh
        if self._shared is None:
            self._shared = fresh
        return self._shared

    def empty(self, nbytes, *, dtype, device):
        return torch.zeros(nbytes, dtype=dtype)

    def rendezvous(self, buf, group):
        handle = _FakeHandle(self.rank, self.world_size, buf, self._pads())
        self.handles.append(handle)
        return handle

    def get_backend(self, device):
        return "FAKE"


@pytest.fixture
def fake_symm(gloo_world, monkeypatch):
    """Preconditions and device syncs stubbed, the symmetric-memory module replaced by the CPU
    fake, the live-pad registry fresh: ``_allocate`` itself runs on the one-rank gloo group."""
    fake = _FakeSymmMem()
    monkeypatch.setattr(symm_mem_pool, "_symm_mem", lambda: fake)
    monkeypatch.setattr(symm_mem_pool, "_local_preconditions", lambda group, device: [])
    monkeypatch.setattr(symm_mem_pool.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(symm_mem_pool, "_live_pads", {})
    return fake


def test_allocate_registers_the_live_pad_and_free_removes_it(gloo_world, fake_symm):
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.signal_pad_ptrs == ()
    with pytest.raises(RuntimeError, match=r"reserve\(\) first"):
        pool.local_signal_pad()
    assert pool.reserve(4096, slots=2, min_channels=4) is True
    handle = fake_symm.handles[-1]
    pad_ptr = handle.signal_pad_ptrs[0]
    assert pool.signal_pad_ptrs == (pad_ptr,)
    assert symm_mem_pool._live_pads == {pad_ptr: repr(pool)}
    assert f"pad={pad_ptr:#x}" in repr(pool)
    # The accessor is this rank's data-channel words: int32, channels x world_size of them, at
    # the pad, sharing its memory; the pad was zeroed by the allocation.
    view = pool.local_signal_pad()
    assert view.dtype == torch.int32 and view.shape == (pool.channels * pool.world_size,)
    assert view.data_ptr() == pad_ptr and pool.channels == 9216 // 4 - 1
    assert not view.any()
    handle.pads[0].view(torch.int32)[5] = 1
    assert view[5] == 1
    pool.release()
    assert symm_mem_pool._live_pads == {} and pool.signal_pad_ptrs == ()
    pool.release()  # idempotent, nothing left to unregister


def test_second_pool_on_a_live_pad_is_refused_before_its_flags_are_zeroed(gloo_world, fake_symm):
    fake_symm.shared_pad = True
    first = SymmMemPool(gloo_world, "cpu", timeout_ms=1234)
    assert first.reserve(4096) is True
    pad_words = fake_symm.handles[-1].pads[0].view(torch.int32)
    pad_words[7] = 1  # a live flag of the first pool (a TP pool idles with its consumed flag set)
    second = SymmMemPool(gloo_world, "cpu")
    with pytest.raises(
        RuntimeError, match=r"signal pad at 0x[0-9a-f]+ .* is live under SymmMemPool\("
    ) as info:
        second.reserve(4096)
    assert "timeout_ms=1234" in str(info.value)  # the holder is named
    assert "set_signal_pad_size" in str(info.value) and "no_split" in str(info.value)
    assert not second.reserved and second.signal_pad_ptrs == ()
    assert pad_words[7] == 1  # the holder's flag survived: nothing was zeroed
    assert list(symm_mem_pool._live_pads) == [first.signal_pad_ptrs[0]]
    # The probe's tiny rendezvous lands on the same pad and declines with the reason.
    ok, reason = probe_symmetric_memory(gloo_world, "cpu")
    assert ok is False
    assert "tiny symmetric rendezvous failed: RuntimeError" in reason and "is live under" in reason
    assert pad_words[7] == 1 and list(symm_mem_pool._live_pads) == [first.signal_pad_ptrs[0]]
    # Releasing the holder frees the pad: the probe passes and the second pool reserves on it.
    first.release()
    assert symm_mem_pool._live_pads == {}
    assert probe_symmetric_memory(gloo_world, "cpu") == (True, "")
    assert symm_mem_pool._live_pads == {}  # the probe released its own allocation
    assert second.reserve(4096) is True
    assert symm_mem_pool._live_pads == {second.signal_pad_ptrs[0]: repr(second)}
    assert pad_words[7] == 0  # zeroed by the new holder, as every allocation does
    second.release()
    assert symm_mem_pool._live_pads == {}


@pytest.fixture
def four_rank_pool(gloo_world, fake_symm):
    """A reserved pool that believes it is rank 2 of 4 (the gloo group has one rank; the fake
    handle carries four pads), so the flag-word arithmetic is checked off the trivial case."""
    pool = SymmMemPool(gloo_world, "cpu")
    pool.rank, pool.world_size = 2, 4
    fake_symm.rank, fake_symm.world_size = 2, 4
    pool._allocate(PoolLayout(4, 1, 256))
    assert pool.channels == 9216 // 16 - 1 and len(pool.signal_pad_ptrs) == 4
    return pool


def test_local_signal_pad_is_this_ranks_data_channel_words(four_rank_pool):
    pool = four_rank_pool
    view = pool.local_signal_pad()
    assert view.dtype == torch.int32 and view.shape == (pool.channels * 4,)
    assert view.data_ptr() == pool.signal_pad_ptrs[2]
    assert len(set(pool.signal_pad_ptrs)) == 4  # one pad per rank, as mapped into this process


def test_signal_write_eager_is_one_stream_memory_write_of_the_put_word(
    four_rank_pool, fake_symm, monkeypatch
):
    pool = four_rank_pool
    writes: list[tuple] = []
    result = [0]

    def cu_stream_write_value32(stream, addr, value, flags):
        writes.append((stream, addr, value, flags))
        return (result[0],)

    monkeypatch.setattr(
        symm_mem_pool,
        "_cuda_driver",
        lambda: SimpleNamespace(cuStreamWriteValue32=cu_stream_write_value32),
    )
    monkeypatch.setattr(symm_mem_pool, "_is_capturing", lambda: False)

    def no_switch(stream):
        raise AssertionError("the eager path must not switch streams")

    monkeypatch.setattr(symm_mem_pool.torch.cuda, "stream", no_switch)
    stream = SimpleNamespace(cuda_stream=0x1234)
    pool.signal_write(3, 5, stream)
    word = 4 * 5 + 2  # world_size * channel + rank: the word put_signal would write
    assert writes == [(0x1234, pool.signal_pad_ptrs[3] + 4 * word, 1, 0)]
    assert fake_symm.memsets == []
    # Checks run before any write.
    with pytest.raises(ValueError, match="rank 4 out of range"):
        pool.signal_write(4, 0, stream)
    with pytest.raises(ValueError, match="channel .* out of range"):
        pool.signal_write(0, pool.channels, stream)
    assert len(writes) == 1
    # A driver error is raised with its code, never swallowed.
    result[0] = 1
    with pytest.raises(RuntimeError, match=r"cuStreamWriteValue32 failed \(1; peer 0, channel 0"):
        pool.signal_write(0, 0, stream)
    pool._free(fence=False)
    with pytest.raises(RuntimeError, match=r"reserve\(\) first"):
        pool.signal_write(0, 0, stream)


def test_signal_write_under_capture_is_a_memset_node_on_the_forked_stream(
    four_rank_pool, fake_symm, monkeypatch
):
    pool = four_rank_pool

    def no_driver():
        raise AssertionError("the capture path must not call the driver")

    monkeypatch.setattr(symm_mem_pool, "_cuda_driver", no_driver)
    entered: list = []

    @contextlib.contextmanager
    def fake_stream_context(stream):
        entered.append(stream)
        yield

    monkeypatch.setattr(symm_mem_pool.torch.cuda, "stream", fake_stream_context)
    capturing = iter([True, True])  # the current stream captures, and so does the forked one
    monkeypatch.setattr(symm_mem_pool, "_is_capturing", lambda: next(capturing))
    stream = SimpleNamespace(cuda_stream=0x1234)
    pool.signal_write(1, 7, stream)
    word = 4 * 7 + 2
    assert entered == [stream]
    assert fake_symm.memsets == [(pool.signal_pad_ptrs[1], torch.uint32, word, 1, 1)]
    assert pool._peer_signal_pad(1) is pool._peer_signal_pad(1)  # cached per allocation
    # A stream outside the capture is refused instead of writing the flag at capture time.
    capturing = iter([True, False])
    with pytest.raises(RuntimeError, match="outside the capture"):
        pool.signal_write(1, 7, stream)
    assert len(fake_symm.memsets) == 1 and entered == [stream, stream]
