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
"""GPU tests for the bounce_v2 C++ mechanism bindings.

Unlike the pure-logic suites next door (which run without the compiled
tensorrt_llm wheel), this module exercises the REAL bindings:
FabricArena / BatchedCopyPool / CompletionPoller / NixlTransferAgent
(register_region / post_transfer_1to1). It therefore requires an installed
tensorrt_llm wheel with the transfer-agent binding AND a CUDA device, and
skips itself entirely when either is missing.
"""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("bounce_v2 mechanism tests require a CUDA device", allow_module_level=True)

tab = pytest.importorskip(
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
    reason="bounce_v2 mechanism tests require the compiled tensorrt_llm wheel",
)

DEVICE = 0
KIB = 1 << 10
MIB = 1 << 20
# The C++ plan builder splits runs into <=64 KiB pieces (kCopySplitBytes).
COPY_SPLIT_BYTES = 64 * KIB


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _u64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64)


def _u32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint32)


def _rand_bytes(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 256, (max(n, 1),), dtype=torch.uint8, generator=gen)[:n].cuda()


def _collect(poller, timeout_ms: int, got: dict[int, tuple[int, int]]) -> None:
    """Drain once and merge the rows into `got`, asserting no id is reported twice."""
    for cid, kind, ok in poller.drain(timeout_ms).tolist():
        assert cid not in got, f"completion id {cid} reported twice"
        got[cid] = (kind, ok)


def _drain_ids(
    poller, expected_ids, deadline_s: float = 30.0, got=None
) -> dict[int, tuple[int, int]]:
    """Drain until every id in expected_ids is seen (bounded). Returns {id: (kind, ok)}.
    `got` carries completions already drained (e.g. during a BUSY-retry loop).
    """
    expected = set(expected_ids)
    got = dict(got or {})
    deadline = time.monotonic() + deadline_s
    while not expected <= set(got):
        assert time.monotonic() < deadline, (
            f"timed out draining completions; got {sorted(got)}, want {sorted(expected)}"
        )
        _collect(poller, 200, got)
    return got


def _submit_retry(pool, poller, srcs, dsts, sizes, got, deadline_s: float = 30.0) -> int:
    """submit_copy with a bounded BUSY-retry loop; completions drained while waiting for a
    free context are merged into `got` (they belong to earlier submissions).
    """
    deadline = time.monotonic() + deadline_s
    while True:
        cid = pool.submit_copy(_u64(srcs), _u64(dsts), _u32(sizes))
        if cid != tab.BatchedCopyPool.BUSY:
            return cid
        assert time.monotonic() < deadline, "pool stayed BUSY past the deadline"
        # Recycling happens on the poll thread; drain doubles as a bounded wait.
        _collect(poller, 50, got)


def _submit_and_wait(pool, poller, srcs, dsts, sizes, deadline_s: float = 30.0) -> None:
    """submit_copy with BUSY retry, then wait for its ok=1 completion (bounded)."""
    got: dict[int, tuple[int, int]] = {}
    cid = _submit_retry(pool, poller, srcs, dsts, sizes, got, deadline_s)
    kind, ok = _drain_ids(poller, [cid], deadline_s, got)[cid]
    assert kind == tab.CompletionPoller.KIND_EVENT
    assert ok == 1


def _roundtrip(pool, poller, sizes: list[int], seed: int) -> None:
    """Gather scattered source regions into a packed staging run, scatter them back out to a
    separate destination buffer, and require byte-exact regions AND untouched inter-region gaps.
    """
    gap = 160  # deliberately not a multiple of 32 so region starts are unaligned
    total = int(sum(sizes))
    spread = int(sum(s + gap for s in sizes)) + 256

    src_buf = _rand_bytes(spread, seed)
    stage = torch.zeros(max(total, 1), dtype=torch.uint8, device="cuda")
    dst_buf = torch.zeros(spread, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    offsets, off = [], 3  # start at an odd offset: nothing here may rely on alignment
    for s in sizes:
        offsets.append(off)
        off += s + gap
    packed, p = [], 0
    for s in sizes:
        packed.append(p)
        p += s

    # gather: scattered src regions -> one packed staging run
    _submit_and_wait(
        pool,
        poller,
        [src_buf.data_ptr() + o for o in offsets],
        [stage.data_ptr() + q for q in packed],
        sizes,
    )
    # scatter: the packed run -> scattered regions of a fresh destination
    _submit_and_wait(
        pool,
        poller,
        [stage.data_ptr() + q for q in packed],
        [dst_buf.data_ptr() + o for o in offsets],
        sizes,
    )

    torch.cuda.synchronize()
    src_cpu, dst_cpu = src_buf.cpu(), dst_buf.cpu()
    covered = torch.zeros(spread, dtype=torch.bool)
    for o, s in zip(offsets, sizes):
        assert torch.equal(dst_cpu[o : o + s], src_cpu[o : o + s]), f"region @{o} size {s} differs"
        covered[o : o + s] = True
    assert not dst_cpu[~covered].any(), "scatter wrote outside its target regions"


# --------------------------------------------------------------------------------------------
# fixtures (fresh objects per test; teardown in reverse order)
# --------------------------------------------------------------------------------------------
@pytest.fixture
def poller():
    p = tab.CompletionPoller(poll_interval_us=50)
    yield p
    p.shutdown()


@pytest.fixture
def pool(poller):
    # keep_alive in the binding ties the poller's lifetime to the pool; fixture order
    # (poller torn down after pool) matches the required destruction order anyway.
    p = tab.BatchedCopyPool(num_streams=2, max_plan_entries=4096, device_id=DEVICE, poller=poller)
    yield p
    torch.cuda.synchronize()  # no in-flight kernel may outlive the pool's pinned plan buffers


def _make_agent(tag: str):
    name = f"bounce_v2_test_{tag}_{uuid.uuid4().hex[:8]}"
    return name, tab.NixlTransferAgent(tab.BaseAgentConfig(name))


@pytest.fixture
def agent_pair():
    src_name, src_agent = _make_agent("src")
    dst_name, dst_agent = _make_agent("dst")
    try:
        yield (src_name, src_agent), (dst_name, dst_agent)
    finally:
        src_agent.shutdown()
        dst_agent.shutdown()


# --------------------------------------------------------------------------------------------
# 1. FabricArena
# --------------------------------------------------------------------------------------------
class TestFabricArena:
    def test_construction_size_and_alignment(self):
        nbytes = 4 * MIB + 12345  # deliberately not a round size
        arena = tab.FabricArena(nbytes, DEVICE, require_fabric=False)
        assert arena.size == nbytes
        assert arena.base_ptr != 0
        # Both backing paths (cudaMalloc and fabric/VMM) guarantee >=256B alignment.
        assert arena.base_ptr % 256 == 0
        assert isinstance(arena.is_fabric, bool)

    def test_require_fabric_contract(self):
        probe = tab.FabricArena(KIB, DEVICE, require_fabric=False)
        if probe.is_fabric:
            arena = tab.FabricArena(KIB, DEVICE, require_fabric=True)
            assert arena.is_fabric
        else:
            # Non-fabric box (e.g. x86 CI): require_fabric=True must raise, not fall back.
            with pytest.raises(RuntimeError, match="fabric"):
                tab.FabricArena(KIB, DEVICE, require_fabric=True)

    def test_repeated_construction_destruction(self):
        seen = set()
        for _ in range(4):
            arena = tab.FabricArena(8 * MIB, DEVICE, require_fabric=False)
            assert arena.size == 8 * MIB
            seen.add(arena.base_ptr)
            del arena
        # Two live at once must not overlap.
        a = tab.FabricArena(MIB, DEVICE, require_fabric=False)
        b = tab.FabricArena(MIB, DEVICE, require_fabric=False)
        assert a.base_ptr + a.size <= b.base_ptr or b.base_ptr + b.size <= a.base_ptr
        assert len(seen) >= 1  # addresses may be recycled; construction itself must be clean


# --------------------------------------------------------------------------------------------
# 2. BatchedCopyPool + CompletionPoller happy path
# --------------------------------------------------------------------------------------------
class TestBatchedCopy:
    def test_roundtrip_mixed_sizes(self, pool, poller):
        # unaligned (1/3/17), 32B-aligned, and large (>1 MiB, exercises 64KiB run-splitting)
        sizes = [1, 3, 17, 32, 4 * KIB, 32 * KIB, MIB + 5, 2 * MIB]
        _roundtrip(pool, poller, sizes, seed=1234)

    def test_roundtrip_many_descs(self, pool, poller):
        # 2048 raw descs with max_plan_entries=4096: mixed tiny sizes plus a sprinkle of
        # >64KiB entries so the plan builder must split under a tight remaining budget.
        rng = np.random.default_rng(4321)
        sizes = [int(rng.integers(1, 97)) for _ in range(2048)]
        for i in range(0, 2048, 128):
            sizes[i] = 96 * KIB  # > kCopySplitBytes: forces splitting inside a big plan
        _roundtrip(pool, poller, sizes, seed=4321)

    def test_roundtrip_large_runs_split(self, pool, poller):
        # Few huge runs: each must be split into ~16 pieces of 64KiB internally.
        sizes = [MIB] * 8
        assert sizes[0] > COPY_SPLIT_BYTES
        _roundtrip(pool, poller, sizes, seed=99)

    def test_zero_total(self, pool, poller):
        cid = pool.submit_copy(_u64([]), _u64([]), _u32([]))
        assert cid != tab.BatchedCopyPool.BUSY
        kind, ok = _drain_ids(poller, [cid])[cid]
        assert (kind, ok) == (tab.CompletionPoller.KIND_EVENT, 1)

    def test_contexts_recycle(self, pool, poller):
        initial = pool.free_count()
        assert initial == pool.num_streams == 2
        buf = torch.zeros(4 * KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        ids, got = [], {}
        for _ in range(6):  # more submissions than streams: recycling must happen mid-loop
            ids.append(
                _submit_retry(
                    pool, poller, [buf.data_ptr()], [buf.data_ptr() + 2 * KIB], [KIB], got
                )
            )
        got = _drain_ids(poller, ids, got=got)
        assert all(got[i] == (tab.CompletionPoller.KIND_EVENT, 1) for i in ids)
        # onTerminal returns the context BEFORE the completion is published, so by the
        # time drain handed us every completion the free list is full again.
        assert pool.free_count() == initial

    def test_plan_overflow_raises(self, poller):
        small = tab.BatchedCopyPool(
            num_streams=1, max_plan_entries=4, device_id=DEVICE, poller=poller
        )
        buf = torch.zeros(KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        n = 5  # raw n > maxPlanEntries must throw (not BUSY, not truncate)
        with pytest.raises(RuntimeError):
            small.submit_copy(
                _u64([buf.data_ptr()] * n), _u64([buf.data_ptr() + 512] * n), _u32([1] * n)
            )
        assert small.free_count() == 1  # the context was returned on the failure path
        torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------
# 3. BUSY backpressure
# --------------------------------------------------------------------------------------------
class TestBusyBackpressure:
    def test_busy_then_recovers(self):
        # A slow (20 ms) poller so a context CANNOT be recycled in the microseconds between
        # our submits, plus 128 MiB copies so the events cannot have fired yet either.
        poller = tab.CompletionPoller(poll_interval_us=20_000)
        try:
            pool = tab.BatchedCopyPool(
                num_streams=2, max_plan_entries=4096, device_id=DEVICE, poller=poller
            )
            big = 128 * MIB
            src = torch.zeros(big, dtype=torch.uint8, device="cuda")
            dst = torch.zeros(big, dtype=torch.uint8, device="cuda")
            small = torch.zeros(KIB, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()

            ids = [
                pool.submit_copy(_u64([src.data_ptr()]), _u64([dst.data_ptr()]), _u32([big]))
                for _ in range(pool.num_streams)
            ]
            assert all(i != tab.BatchedCopyPool.BUSY for i in ids)
            assert pool.free_count() == 0

            busy = pool.submit_copy(
                _u64([small.data_ptr()]), _u64([small.data_ptr() + 512]), _u32([256])
            )
            assert busy == tab.BatchedCopyPool.BUSY

            got = _drain_ids(poller, ids)
            assert all(got[i] == (tab.CompletionPoller.KIND_EVENT, 1) for i in ids)
            assert pool.free_count() == pool.num_streams

            retry = pool.submit_copy(
                _u64([small.data_ptr()]), _u64([small.data_ptr() + 512]), _u32([256])
            )
            assert retry != tab.BatchedCopyPool.BUSY
            assert _drain_ids(poller, [retry])[retry] == (tab.CompletionPoller.KIND_EVENT, 1)
            torch.cuda.synchronize()
        finally:
            poller.shutdown()


# --------------------------------------------------------------------------------------------
# 4. drain semantics
# --------------------------------------------------------------------------------------------
class TestDrainSemantics:
    def test_drain_empty_nonblocking_and_timed(self, poller):
        out = poller.drain(0)
        assert out.shape == (0, 3) and out.dtype == np.int64
        t0 = time.monotonic()
        out = poller.drain(100)  # nothing pending: must return empty after <= ~timeout
        assert out.shape == (0, 3)
        assert time.monotonic() - t0 < 5.0

    def test_ids_unique_monotonic_and_batched(self, pool, poller):
        buf = torch.zeros(8 * KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        ids, got = [], {}
        for _ in range(4):
            ids.append(
                _submit_retry(
                    pool, poller, [buf.data_ptr()], [buf.data_ptr() + 4 * KIB], [KIB], got
                )
            )
        assert len(set(ids)) == len(ids)
        assert ids == sorted(ids)  # assigned monotonically in registration order
        got = _drain_ids(poller, ids, got=got)
        assert set(got) == set(ids)  # nothing else was pending on this fresh poller
        for _, (kind, ok) in got.items():
            assert (kind, ok) == (tab.CompletionPoller.KIND_EVENT, 1)


# --------------------------------------------------------------------------------------------
# 5. post_transfer_1to1 over real NIXL
# --------------------------------------------------------------------------------------------
class TestNixlTransfer:
    def test_arena_to_arena_write_roundtrip(self, poller, pool, agent_pair):
        (_, src_agent), (dst_name, dst_agent) = agent_pair
        n = 256 * KIB + 7  # odd size: single-descriptor writes must not assume alignment
        src_arena = tab.FabricArena(MIB, DEVICE, require_fabric=False)
        dst_arena = tab.FabricArena(MIB, DEVICE, require_fabric=False)
        try:
            assert src_agent.register_region(src_arena.base_ptr, src_arena.size, DEVICE)
            assert dst_agent.register_region(dst_arena.base_ptr, dst_arena.size, DEVICE)
            # One-directional, like the real sender: only src loads the receiver's desc.
            src_agent.load_remote_agent(dst_name, dst_agent.get_local_agent_desc())

            payload = _rand_bytes(n, seed=2026)
            torch.cuda.synchronize()
            _submit_and_wait(pool, poller, [payload.data_ptr()], [src_arena.base_ptr], [n])

            xid = src_agent.post_transfer_1to1(
                src_arena.base_ptr, dst_arena.base_ptr, n, DEVICE, DEVICE, dst_name, poller
            )
            assert xid >= 0
            kind, ok = _drain_ids(poller, [xid])[xid]
            assert (kind, ok) == (tab.CompletionPoller.KIND_XFER, 1)

            back = torch.zeros(n, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            _submit_and_wait(pool, poller, [dst_arena.base_ptr], [back.data_ptr()], [n])
            torch.cuda.synchronize()
            assert torch.equal(payload, back)
        finally:
            torch.cuda.synchronize()
            src_agent.deregister_region(src_arena.base_ptr, src_arena.size, DEVICE)
            dst_agent.deregister_region(dst_arena.base_ptr, dst_arena.size, DEVICE)

    def test_post_to_unknown_peer_fails_fast(self, poller, agent_pair):
        (_, src_agent), _ = agent_pair
        arena = tab.FabricArena(MIB, DEVICE, require_fabric=False)
        try:
            assert src_agent.register_region(arena.base_ptr, arena.size, DEVICE)
            t0 = time.monotonic()
            res = src_agent.post_transfer_1to1(
                arena.base_ptr,
                arena.base_ptr + 64 * KIB,
                KIB,
                DEVICE,
                DEVICE,
                "bounce_v2_test_never_loaded_peer",
                poller,
            )
            elapsed = time.monotonic() - t0
            # Never-loaded peer: the post itself must fail with the negative sentinel —
            # bounded (no hang), with nothing left pending on the poller.
            assert res == -1
            assert elapsed < 30.0
            assert poller.drain(0).shape == (0, 3)
        finally:
            src_agent.deregister_region(arena.base_ptr, arena.size, DEVICE)


# --------------------------------------------------------------------------------------------
# 6. poller shutdown
# --------------------------------------------------------------------------------------------
class TestPollerShutdown:
    def test_shutdown_with_inflight_event(self):
        # Slow poller + big copy: the event is (near-)certainly still pending at shutdown.
        poller = tab.CompletionPoller(poll_interval_us=50_000)
        pool = tab.BatchedCopyPool(
            num_streams=1, max_plan_entries=4096, device_id=DEVICE, poller=poller
        )
        src = torch.zeros(64 * MIB, dtype=torch.uint8, device="cuda")
        dst = torch.zeros(64 * MIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        cid = pool.submit_copy(_u64([src.data_ptr()]), _u64([dst.data_ptr()]), _u32([64 * MIB]))
        assert cid != tab.BatchedCopyPool.BUSY

        poller.shutdown()  # must neither crash nor hang with the event in flight

        # Defined post-shutdown behavior: the pending entry was terminated and is drainable
        # exactly once (ok=0 if it was still pending, ok=1 if a sweep beat the shutdown).
        rows = poller.drain(0).tolist()
        assert [r[0] for r in rows] == [cid]
        assert rows[0][1] == tab.CompletionPoller.KIND_EVENT
        assert rows[0][2] in (0, 1)
        # Post-shutdown drains never block and return empty.
        t0 = time.monotonic()
        assert poller.drain(1000).shape == (0, 3)
        assert time.monotonic() - t0 < 5.0
        poller.shutdown()  # idempotent

        # shutdown() ran the event's onTerminal, so the context is already back.
        assert pool.free_count() == 1
        # LOOP1-FINDING: after CompletionPoller.shutdown() with an in-flight copy, the pool's
        # context is released (onTerminal) while its kernel may still be running. The
        # BatchedCopyPool destructor only waits for the free list to fill, so it can
        # cudaFreeHost the pinned plan buffer the live kernel is still reading (device
        # use-after-free). The test synchronizes before dropping the pool to stay on the
        # supported teardown order; the hazard is in the implementation's dtor gating.
        torch.cuda.synchronize()
        del pool

    def test_shutdown_idempotent_when_idle(self, pool, poller):
        poller.shutdown()
        poller.shutdown()
        assert poller.drain(100).shape == (0, 3)
        # Submitting after shutdown must still resolve the id (immediately, ok=0).
        buf = torch.zeros(KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        cid = pool.submit_copy(_u64([buf.data_ptr()]), _u64([buf.data_ptr() + 512]), _u32([16]))
        assert cid != tab.BatchedCopyPool.BUSY
        rows = poller.drain(0).tolist()
        assert rows == [[cid, tab.CompletionPoller.KIND_EVENT, 0]]
        torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------
# 7. register_region / deregister_region
# --------------------------------------------------------------------------------------------
class TestRegisterRegion:
    def test_register_deregister_reregister_cycle(self, agent_pair):
        (_, agent), _ = agent_pair
        arena = tab.FabricArena(2 * MIB, DEVICE, require_fabric=False)
        assert agent.register_region(arena.base_ptr, arena.size, DEVICE) is True
        agent.deregister_region(arena.base_ptr, arena.size, DEVICE)
        assert agent.register_region(arena.base_ptr, arena.size, DEVICE) is True
        agent.deregister_region(arena.base_ptr, arena.size, DEVICE)

    def test_register_nonsense_returns_false(self, agent_pair):
        (_, agent), _ = agent_pair
        # Implemented contract: registerRegionImpl logs a warning and returns False on
        # backend failure — it must not throw for a bogus (non-CUDA) address.
        assert agent.register_region(0xDEAD0000, 4 * KIB, DEVICE) is False

    def test_deregister_unknown_range_does_not_throw(self, agent_pair):
        (_, agent), _ = agent_pair
        arena = tab.FabricArena(MIB, DEVICE, require_fabric=False)
        # deregisterRegionImpl is warn-only by contract: never registered -> no exception.
        agent.deregister_region(arena.base_ptr, arena.size, DEVICE)
