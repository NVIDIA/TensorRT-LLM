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

from tensorrt_llm._torch.disaggregation.bounce_v2.plan import SCATTER_RUN_DTYPE  # noqa: E402

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
    """Drain until every id in expected_ids is seen (bounded).

    Returns {id: (kind, ok)}. `got` carries completions already drained
    (e.g. during a BUSY-retry loop).
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
    """submit_copy with a bounded BUSY-retry loop.

    Completions drained while waiting for a free context are merged into
    `got` (they belong to earlier submissions).
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
    """Gather scattered regions into a packed run, then scatter them back out.

    The scatter targets a separate destination buffer; require byte-exact
    regions AND untouched inter-region gaps.
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
        # LOOP1-FINDING (fixed in round 3): the dtor now unregisters still-pending events
        # from a live poller and destroyContexts() stream-synchronizes before freeing the
        # pinned plan buffers — see test_pool_destruction_with_inflight_context_and_live_
        # poller_is_bounded for the live-poller teardown path. Here shutdown() already
        # cudaEventSynchronize'd the entry, so the kernel is done; the synchronize below is
        # belt-and-braces for the supported teardown order.
        torch.cuda.synchronize()
        del pool

    def test_pool_destruction_with_inflight_context_and_live_poller_is_bounded(self):
        """Round-3 fix for the LOOP1-FINDING: ~BatchedCopyPool detaches safely.

        A slow-sweep poller (first sweep at thread start, next one ~6 s later)
        keeps the submitted copy's event registered well past teardown, so the
        pool is destroyed with the context still in flight and the poller
        LIVE. The fixed destructor waits at most ~2 s, then unregisters its
        pending events from the poller before destroying them; the
        unregistered id then simply never reports. The old destructor waited
        until the free list filled (here: the ~6 s sweep; up to 10 s) and the
        swept id WAS published — both observably different from Python.
        """
        poller = tab.CompletionPoller(poll_interval_us=6_000_000)
        try:
            pool = tab.BatchedCopyPool(
                num_streams=1, max_plan_entries=4096, device_id=DEVICE, poller=poller
            )
            buf = torch.zeros(MIB, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            cid = pool.submit_copy(
                _u64([buf.data_ptr()]), _u64([buf.data_ptr() + 512 * KIB]), _u32([256 * KIB])
            )
            assert cid != tab.BatchedCopyPool.BUSY
            # Precondition: no sweep recycled the context yet (the next sweep
            # is ~6 s away; getting here from poller construction takes ms).
            assert pool.free_count() == 0, "sweep beat the teardown; timing precondition broken"

            t0 = time.monotonic()
            del pool  # in-flight context + LIVE poller: the fixed dtor path
            elapsed = time.monotonic() - t0
            assert elapsed < 4.5, (
                f"pool destruction took {elapsed:.1f}s with an in-flight context "
                f"(old dtor waited for the poller sweep / its 10 s cap)"
            )
        finally:
            poller.shutdown()  # joins once the poll thread's sleep slice ends
        # The unregistered event must never report: neither the shutdown
        # terminal batch nor any later drain may carry its id (the old dtor
        # left it registered and the sweep/shutdown published it).
        assert cid not in [r[0] for r in poller.drain(0).tolist()]
        assert poller.drain(0).shape == (0, 3)
        torch.cuda.synchronize()

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


# --------------------------------------------------------------------------------------------
# 8. submit_scatter_runs (receiver DATA sink: validate + expand + launch in C++)
# --------------------------------------------------------------------------------------------
def _scatter_runs(entries) -> np.ndarray:
    """Build a SCATTER_RUN_DTYPE array.

    Entries are (b_off, dst, d_stride, b_stride, piece, count) tuples in
    the wire field order.
    """
    runs = np.zeros(len(entries), dtype=SCATTER_RUN_DTYPE)
    for i, e in enumerate(entries):
        runs[i] = e
    return runs


def _expand_runs_python(region_base: int, runs: np.ndarray):
    """Local CPU reference expansion of the wire runs.

    Test-only: production has no Python expansion path (the C++ sink is
    required).
    """
    srcs, dsts, sizes = [], [], []
    for r in runs:
        for p in range(int(r["count"])):
            srcs.append(region_base + int(r["bounce_offset"]) + p * int(r["bounce_stride"]))
            dsts.append(int(r["dst_addr"]) + p * int(r["dst_stride"]))
            sizes.append(int(r["piece_size"]))
    return _u64(srcs), _u64(dsts), _u32(sizes)


@pytest.mark.skipif(
    not hasattr(tab.BatchedCopyPool, "submit_scatter_runs"),
    reason="installed transfer-agent binding predates BatchedCopyPool.submit_scatter_runs",
)
class TestSubmitScatterRuns:
    def test_happy_path_matches_python_expansion_byte_exact(self, pool, poller):
        """T4a: the C++ sink scatters exactly like the Python expansion.

        Same raw wire runs through submit_scatter_runs (dst A) and through
        the reference expansion + submit_copy (dst B): both destinations must
        be byte-identical to each other AND to a CPU-computed reference,
        including untouched bytes between the pieces.
        """
        region_bytes = 64 * KIB
        region = _rand_bytes(region_bytes, seed=8141)
        spread = 96 * KIB
        dst_a = torch.zeros(spread, dtype=torch.uint8, device="cuda")
        dst_b = torch.zeros(spread, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()

        def entries_for(dst):
            # Odd offsets/sizes and unequal strides: nothing may assume
            # alignment or dst_stride == bounce_stride.
            return [
                (3, dst.data_ptr() + 5, 8 * KIB + 7, 4 * KIB + 3, 1000, 5),
                (40 * KIB + 1, dst.data_ptr() + 60 * KIB, 3 * KIB, 2 * KIB, 2 * KIB, 3),
                (60 * KIB, dst.data_ptr() + 90 * KIB, 0, 0, 17, 1),  # single piece, strides 0
            ]

        runs_a = _scatter_runs(entries_for(dst_a))
        cid = pool.submit_scatter_runs(
            region.data_ptr(), region_bytes, np.ascontiguousarray(runs_a).view(np.uint8)
        )
        assert cid not in (tab.BatchedCopyPool.BUSY, tab.BatchedCopyPool.SCATTER_REJECTED)
        kind, ok = _drain_ids(poller, [cid])[cid]
        assert (kind, ok) == (tab.CompletionPoller.KIND_EVENT, 1)

        runs_b = _scatter_runs(entries_for(dst_b))
        srcs, dsts, sizes = _expand_runs_python(region.data_ptr(), runs_b)
        _submit_and_wait(pool, poller, srcs, dsts, sizes)

        torch.cuda.synchronize()
        assert torch.equal(dst_a, dst_b), "C++ sink scattered differently from the Python path"
        # And against a CPU reference (also proves the gaps stayed untouched).
        ref = torch.zeros(spread, dtype=torch.uint8)
        region_cpu = region.cpu()
        for b_off, dst_ptr, d_stride, b_stride, piece, count in entries_for(dst_a):
            d_off = dst_ptr - dst_a.data_ptr()
            for p in range(count):
                ref[d_off + p * d_stride : d_off + p * d_stride + piece] = region_cpu[
                    b_off + p * b_stride : b_off + p * b_stride + piece
                ]
        assert torch.equal(dst_a.cpu(), ref)

    @pytest.mark.parametrize("bad", ["offset_past_region", "span_overflow", "strided_overflow"])
    def test_out_of_region_runs_rejected(self, pool, poller, bad):
        """T4b: validation failures return SCATTER_REJECTED and touch nothing.

        Offset beyond region_bytes / span overflow at the tail / a strided
        span walking past the region: -2, no completion row is ever
        published, the destination stays untouched, and no stream context is
        consumed.
        """
        region_bytes = 64 * KIB
        region = _rand_bytes(region_bytes, seed=8142)
        dst = torch.zeros(128 * KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        entry = {
            "offset_past_region": (region_bytes + 1, dst.data_ptr(), 0, 0, 16, 1),
            "span_overflow": (region_bytes - 8, dst.data_ptr(), 0, 0, 16, 1),
            "strided_overflow": (0, dst.data_ptr(), 32 * KIB, 32 * KIB, 4 * KIB, 4),
        }[bad]
        free0 = pool.free_count()
        rc = pool.submit_scatter_runs(
            region.data_ptr(),
            region_bytes,
            np.ascontiguousarray(_scatter_runs([entry])).view(np.uint8),
        )
        assert rc == tab.BatchedCopyPool.SCATTER_REJECTED == -2
        assert poller.drain(0).shape == (0, 3), "a rejected scatter still published a row"
        assert pool.free_count() == free0, "a rejected scatter consumed a stream context"
        torch.cuda.synchronize()
        assert not dst.any(), "a rejected scatter still wrote to the destination"

    def test_busy_then_retry_with_same_raw_runs(self):
        """T4c: BUSY with every context taken; the raw runs retry clean.

        The reactor's backlog contract: the SAME raw runs succeed once a
        context frees. A slow-sweep poller pins the single context in flight
        past the BUSY probe (recycling only happens on the poll sweep).
        """
        poller = tab.CompletionPoller(poll_interval_us=200_000)
        try:
            pool = tab.BatchedCopyPool(
                num_streams=1, max_plan_entries=4096, device_id=DEVICE, poller=poller
            )
            region = _rand_bytes(64 * KIB, seed=8143)
            buf = torch.zeros(8 * MIB, dtype=torch.uint8, device="cuda")
            dst = torch.zeros(8 * KIB, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            occupier = pool.submit_copy(
                _u64([buf.data_ptr()]), _u64([buf.data_ptr() + 4 * MIB]), _u32([4 * MIB])
            )
            assert occupier != tab.BatchedCopyPool.BUSY

            raw = np.ascontiguousarray(
                _scatter_runs([(0, dst.data_ptr(), 4 * KIB, 2 * KIB, 2 * KIB, 2)])
            ).view(np.uint8)
            rc = pool.submit_scatter_runs(region.data_ptr(), 64 * KIB, raw)
            assert rc == tab.BatchedCopyPool.BUSY, "single context was not busy"

            _drain_ids(poller, [occupier])  # sweep recycles the context
            cid = pool.submit_scatter_runs(region.data_ptr(), 64 * KIB, raw)
            assert cid not in (tab.BatchedCopyPool.BUSY, tab.BatchedCopyPool.SCATTER_REJECTED)
            kind, ok = _drain_ids(poller, [cid])[cid]
            assert (kind, ok) == (tab.CompletionPoller.KIND_EVENT, 1)
            torch.cuda.synchronize()
            assert torch.equal(dst[: 2 * KIB], region[: 2 * KIB].to(dst.device))
            assert torch.equal(dst[4 * KIB : 6 * KIB], region[2 * KIB : 4 * KIB])
            del pool
        finally:
            poller.shutdown()

    def test_malformed_blob_length_raises(self, pool):
        """T4d: a runs blob that is not a multiple of 36 bytes raises."""
        region = torch.zeros(4 * KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        with pytest.raises(ValueError):
            pool.submit_scatter_runs(region.data_ptr(), 4 * KIB, np.zeros(35, dtype=np.uint8))
        with pytest.raises(ValueError):
            pool.submit_scatter_runs(region.data_ptr(), 4 * KIB, np.zeros(37, dtype=np.uint8))


# --------------------------------------------------------------------------------------------
# 9. register_plan / launch_chunk (per-request plan handle)
# --------------------------------------------------------------------------------------------
def _launch_retry(pool, poller, handle, chunk_idx, staging_base, got, deadline_s: float = 30.0):
    """launch_chunk with a bounded BUSY-retry loop (mirrors _submit_retry)."""
    deadline = time.monotonic() + deadline_s
    while True:
        cid = pool.launch_chunk(handle, chunk_idx, staging_base)
        if cid != tab.BatchedCopyPool.BUSY:
            return cid
        assert time.monotonic() < deadline, "pool stayed BUSY past the deadline"
        _collect(poller, 50, got)


@pytest.mark.skipif(
    not hasattr(tab.BatchedCopyPool, "register_plan"),
    reason="installed transfer-agent binding predates BatchedCopyPool.register_plan",
)
class TestPlanHandle:
    def test_launch_chunk_matches_submit_copy_expansion_byte_exact(self, pool, poller):
        """P1a: launching from the registered plan == the classic expansion.

        The SAME real plan (built by build_plan, marshalled once via
        flat_gather/register_plan) gathers each chunk to staging A through
        launch_chunk (scalar args) and to staging B through the classic
        submit_copy per-desc expansion: both stagings must be byte-identical
        and match a CPU reference.
        """
        from tensorrt_llm._torch.disaggregation.bounce_v2.plan import build_plan

        n_desc, desc_bytes = 40, 4 * KIB  # 160 KiB over 64 KiB chunks -> 3 chunks
        spread = n_desc * (desc_bytes + 160) + 256
        src = _rand_bytes(spread, seed=4501)
        torch.cuda.synchronize()
        offs = [3 + i * (desc_bytes + 160) for i in range(n_desc)]
        src_ptrs = _u64([src.data_ptr() + o for o in offs])
        dst_ptrs = _u64([0x9000_0000 + i * desc_bytes for i in range(n_desc)])  # gather-unused
        sizes = _u64([desc_bytes] * n_desc)
        plan = build_plan(
            src_ptrs, dst_ptrs, sizes, max_chunk_bytes=64 * KIB, max_descs_per_chunk=4096
        )
        assert plan.num_chunks == 3

        g_srcs, g_offsets, g_sizes, g_starts = plan.flat_gather()
        handle = pool.register_plan(g_srcs, g_offsets, g_sizes, g_starts)
        try:
            for c, chunk in enumerate(plan.chunks):
                stage_a = torch.zeros(chunk.packed_bytes, dtype=torch.uint8, device="cuda")
                stage_b = torch.zeros(chunk.packed_bytes, dtype=torch.uint8, device="cuda")
                torch.cuda.synchronize()
                got: dict[int, tuple[int, int]] = {}
                cid_a = _launch_retry(pool, poller, handle, c, stage_a.data_ptr(), got)
                cid_b = _submit_retry(
                    pool,
                    poller,
                    chunk.src_ptrs,
                    (np.uint64(stage_b.data_ptr()) + chunk.bounce_offsets).astype(np.uint64),
                    chunk.sizes,
                    got,
                )
                rows = _drain_ids(poller, [cid_a, cid_b], got=got)
                assert rows[cid_a] == (tab.CompletionPoller.KIND_EVENT, 1)
                assert rows[cid_b] == (tab.CompletionPoller.KIND_EVENT, 1)
                torch.cuda.synchronize()
                assert torch.equal(stage_a, stage_b), f"chunk {c}: plan-handle gather differs"
                # CPU reference for one desc of the chunk (spot check).
                d0 = int(chunk.bounce_offsets[0])
                s0 = int(chunk.src_ptrs[0]) - src.data_ptr()
                assert torch.equal(
                    stage_a[d0 : d0 + desc_bytes].cpu(), src[s0 : s0 + desc_bytes].cpu()
                )
        finally:
            pool.release_plan(handle)

    def test_boundary_validation_raises(self, pool):
        """P1b: malformed chunk boundaries are rejected at registration."""
        n = 4
        srcs = _u64([0x9000_0000 + i * KIB for i in range(n)])
        offsets = _u64([i * KIB for i in range(n)])
        sizes = _u32([KIB] * n)
        for bad_starts in (
            [0, 3, 2, 4],  # non-monotonic
            [1, 4],  # first boundary != 0
            [0, 3],  # last boundary != n_descs
        ):
            with pytest.raises(ValueError):
                pool.register_plan(srcs, offsets, sizes, _u64(bad_starts))
        # Per-chunk desc count above max_plan_entries (fixture pool: 4096).
        big = int(pool.max_plan_entries) + 1
        with pytest.raises(ValueError):
            pool.register_plan(
                _u64(np.zeros(big)), _u64(np.zeros(big)), _u32(np.zeros(big)), _u64([0, big])
            )

    def test_release_idempotent_and_launch_after_release_raises(self, pool, poller):
        """P1c/P1d: release_plan is idempotent; a later launch_chunk raises.

        The ValueError is the deterministic terminal the reactor's
        launch-error path relies on for a launch racing the failure-side
        release.
        """
        buf = torch.zeros(8 * KIB, dtype=torch.uint8, device="cuda")
        stage = torch.zeros(4 * KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        handle = pool.register_plan(
            _u64([buf.data_ptr()]), _u64([0]), _u32([4 * KIB]), _u64([0, 1])
        )
        got: dict[int, tuple[int, int]] = {}
        cid = _launch_retry(pool, poller, handle, 0, stage.data_ptr(), got)
        assert _drain_ids(poller, [cid], got=got)[cid] == (tab.CompletionPoller.KIND_EVENT, 1)
        # Chunk index out of range on a LIVE handle also raises.
        with pytest.raises(ValueError):
            pool.launch_chunk(handle, 1, stage.data_ptr())
        pool.release_plan(handle)
        pool.release_plan(handle)  # idempotent: no error
        with pytest.raises(ValueError):
            pool.launch_chunk(handle, 0, stage.data_ptr())
        torch.cuda.synchronize()

    def test_busy_when_contexts_exhausted_then_retry(self):
        """P1e: BUSY with every context taken; the same launch retries clean.

        A slow-sweep poller pins the single context in flight past the BUSY
        probe (recycling only happens on the poll sweep).
        """
        poller = tab.CompletionPoller(poll_interval_us=200_000)
        try:
            pool = tab.BatchedCopyPool(
                num_streams=1, max_plan_entries=4096, device_id=DEVICE, poller=poller
            )
            src = _rand_bytes(4 * KIB, seed=4502)
            stage = torch.zeros(4 * KIB, dtype=torch.uint8, device="cuda")
            buf = torch.zeros(8 * MIB, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            handle = pool.register_plan(
                _u64([src.data_ptr()]), _u64([0]), _u32([4 * KIB]), _u64([0, 1])
            )
            occupier = pool.submit_copy(
                _u64([buf.data_ptr()]), _u64([buf.data_ptr() + 4 * MIB]), _u32([4 * MIB])
            )
            assert occupier != tab.BatchedCopyPool.BUSY
            assert pool.launch_chunk(handle, 0, stage.data_ptr()) == tab.BatchedCopyPool.BUSY

            _drain_ids(poller, [occupier])  # sweep recycles the context
            cid = pool.launch_chunk(handle, 0, stage.data_ptr())
            assert cid != tab.BatchedCopyPool.BUSY
            kind, ok = _drain_ids(poller, [cid])[cid]
            assert (kind, ok) == (tab.CompletionPoller.KIND_EVENT, 1)
            torch.cuda.synchronize()
            assert torch.equal(stage, src)
            pool.release_plan(handle)
            del pool
        finally:
            poller.shutdown()
