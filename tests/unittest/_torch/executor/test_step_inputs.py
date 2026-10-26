# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packing a decode step's small input copies into one upload and one scatter launch."""

import threading

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import h2d_cache, step_inputs
from tensorrt_llm._torch.pyexecutor.h2d_cache import H2DSkipCache, copy_h2d_if_changed
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack


def _scatter_op_registered() -> bool:
    return torch.cuda.is_available() and hasattr(torch.ops.trtllm, "step_inputs_scatter")


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not _scatter_op_registered(),
        reason="torch.ops.trtllm.step_inputs_scatter is not registered (built from "
        "cpp/tensorrt_llm/thop/minimaxM3/stepInputsScatterOp.cpp)",
    ),
]

# About half a second of GPU time: long enough that work queued behind it is still pending when
# the host checks, short enough for a unit test.
_SLEEP_CYCLES = 1_000_000_000


def _device():
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def pack():
    """A fresh pack, active for the test body."""
    step_pack = StepInputPack(_device())
    with step_inputs.active(step_pack):
        yield step_pack
        step_inputs.flush()
    torch.cuda.synchronize()


def _host(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype).pin_memory()


def _zeros(*shape, dtype=torch.int32):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


# ---------------------------------------------------------------------------
# No active pack
# ---------------------------------------------------------------------------


def test_without_an_active_pack_every_call_is_a_plain_copy():
    dst = _zeros(4)
    d2d_dst = _zeros(4)
    src_dev = torch.arange(4, dtype=torch.int32, device="cuda")

    step_inputs.begin()
    assert not step_inputs.is_open()
    assert step_inputs.copy_(dst, _host([1, 2, 3, 4])) is dst
    step_inputs.copy_(d2d_dst, src_dev)
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4]
    assert d2d_dst.tolist() == [0, 1, 2, 3]
    step_inputs.flush()
    step_inputs.discard()


def test_an_inactive_pack_is_not_used():
    idle = StepInputPack(_device())
    dst = _zeros(2)

    step_inputs.copy_(dst, _host([3, 4]))
    torch.cuda.synchronize()

    assert dst.tolist() == [3, 4]
    assert not idle.open and idle.staged == []


# ---------------------------------------------------------------------------
# Deferral and ordering
# ---------------------------------------------------------------------------


def test_packed_host_copy_lands_only_at_flush(pack):
    dst = _zeros(4)
    src = _host([1, 2, 3, 4])

    step_inputs.begin()
    assert step_inputs.is_open() and pack.open
    assert step_inputs.copy_(dst, src) is dst
    # The pack snapshots the host bytes at the copy site.
    src.fill_(7)
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0, 0]

    step_inputs.flush()
    assert not step_inputs.is_open()
    assert pack.staged == []
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4]


def test_packed_host_copy_into_a_row_strided_2d_slice(pack):
    table = torch.full((4, 6), -1, dtype=torch.int32, device="cuda")
    dst = table[1:3, 2:5]
    assert not dst.is_contiguous() and dst.stride(1) == 1
    expected = table.clone()
    expected[1:3, 2:5] = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    step_inputs.begin()
    step_inputs.copy_(dst, _host([[1, 2, 3], [4, 5, 6]]))
    torch.cuda.synchronize()
    assert torch.equal(table, torch.full_like(table, -1))

    step_inputs.flush()
    torch.cuda.synchronize()
    assert torch.equal(table, expected)


def test_packed_host_copy_converts_dtype_on_the_host(pack):
    dst = _zeros(3)

    step_inputs.begin()
    step_inputs.copy_(dst, _host([7, -8, 9], dtype=torch.int64))
    step_inputs.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [7, -8, 9]


def test_packed_device_copy_lands_only_at_flush(pack):
    dst = _zeros(5)
    src = torch.arange(5, dtype=torch.int32, device="cuda")

    step_inputs.begin()
    step_inputs.copy_(dst, src)
    torch.cuda.synchronize()
    assert dst.tolist() == [0] * 5

    step_inputs.flush()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 1, 2, 3, 4]


def test_device_copy_reads_what_a_host_copy_of_the_same_pack_wrote(pack):
    """The graph runner copies the engine's freshly uploaded buffers into its static inputs."""
    staged = _zeros(4)
    static = _zeros(1, 8)

    step_inputs.begin()
    step_inputs.copy_(staged, _host([10, 11, 12, 13]))
    step_inputs.copy_(static[:, :4], staged.unsqueeze(0))
    step_inputs.flush()
    torch.cuda.synchronize()

    assert static.tolist() == [[10, 11, 12, 13, 0, 0, 0, 0]]


# The scatter's CTA copies descriptor i with thread t on bytes t, t + 256, ... of it. When the
# second copy starts TAIL bytes before the end of the first (TAIL not a multiple of the warp
# size), its first bytes are the ones the first copy writes last, from other warps: without a
# barrier between descriptors, a warp that finished the first copy early writes (or reads) them
# before a slower warp has written its part of the first copy.
_TAILS = [239, 1000, 4073]


@pytest.mark.parametrize("tail", _TAILS)
@pytest.mark.parametrize("kind", ["host", "device"])
def test_overlapping_destinations_land_in_descriptor_order(pack, kind, tail):
    """The later copy wins on the overlap, as it would eagerly."""
    # Host copies share the 64 KiB staging buffer; device copies are bounded only by memory, and a
    # long first copy lets the warps drift apart.
    n = 24 * 1024 if kind == "host" else 1 << 20
    m = 8 * 1024
    first = torch.full((n,), 1, dtype=torch.uint8)
    second = torch.full((m,), 2, dtype=torch.uint8)
    if kind == "host":
        first, second = first.pin_memory(), second.pin_memory()
    else:
        first, second = first.cuda(), second.cuda()
    start = n - tail
    expected = torch.ones(start + m, dtype=torch.uint8)
    expected[start:] = 2

    mismatches = []
    for trial in range(50):
        buf = torch.zeros(start + m, dtype=torch.uint8, device="cuda")
        step_inputs.begin()
        step_inputs.copy_(buf[:n], first)
        step_inputs.copy_(buf[start:], second)
        step_inputs.flush()
        torch.cuda.synchronize()
        wrong = int((buf.cpu() != expected).sum())
        if wrong:
            mismatches.append((trial, wrong))

    assert not mismatches, f"(trial, overlap bytes where the earlier copy won): {mismatches}"


@pytest.mark.parametrize("tail", _TAILS)
def test_a_device_copy_reads_the_destination_of_an_earlier_device_copy(pack, tail):
    """A chain of device copies: the second reads the bytes the first writes last."""
    n = 1 << 20
    source = torch.randint(1, 256, (n,), dtype=torch.uint8, device="cuda")
    expected = source[n - tail :].clone()

    mismatches = []
    for trial in range(50):
        middle = torch.zeros(n, dtype=torch.uint8, device="cuda")
        out = torch.zeros(tail, dtype=torch.uint8, device="cuda")
        step_inputs.begin()
        step_inputs.copy_(middle, source)
        step_inputs.copy_(out, middle[n - tail :])
        step_inputs.flush()
        torch.cuda.synchronize()
        wrong = int((out != expected).sum())
        if wrong:
            mismatches.append((trial, wrong))

    assert not mismatches, (
        f"(trial, stale bytes read before the first copy wrote them): {mismatches}"
    )


# ---------------------------------------------------------------------------
# Staged tensors stay alive
# ---------------------------------------------------------------------------


def test_a_temporary_device_source_outlives_its_copy(pack):
    """The pack holds the source until the scatter is queued, so its memory is not reused."""
    numel = 1 << 16
    expected = torch.arange(numel, dtype=torch.int32) * 3 + 1
    dst = _zeros(numel)

    step_inputs.begin()
    # The only reference to this source is the one the pack keeps.
    step_inputs.copy_(dst, torch.arange(numel, dtype=torch.int32, device="cuda") * 3 + 1)
    assert len(pack.staged) == 2
    # Churn the allocator with same-sized blocks: a freed source block would be handed out here.
    garbage = [torch.full((numel,), -7, dtype=torch.int32, device="cuda") for _ in range(16)]
    torch.cuda.synchronize()
    step_inputs.flush()
    del garbage
    # Blocks allocated after the flush are ordered behind the scatter on this stream.
    after = [torch.full((numel,), -9, dtype=torch.int32, device="cuda") for _ in range(16)]
    torch.cuda.synchronize()

    assert torch.equal(dst.cpu(), expected)
    del after


def test_a_temporary_destination_view_outlives_its_copy(pack):
    owner = _zeros(8)

    step_inputs.begin()
    step_inputs.copy_(owner.view(2, 4)[1], _host([5, 6, 7, 8]))
    assert len(pack.staged) == 1
    step_inputs.flush()
    torch.cuda.synchronize()

    assert owner.tolist() == [0, 0, 0, 0, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def test_an_unpackable_copy_flushes_first_and_keeps_order(pack):
    buf = _zeros(8)
    other = _zeros(4)
    later = _zeros(2)

    step_inputs.begin()
    step_inputs.copy_(buf, _host([1] * 8))
    step_inputs.copy_(other, _host([3, 3, 3, 3]))
    # A strided 1-D destination cannot be packed: the pack lands first, then this copy.
    step_inputs.copy_(buf[::2], _host([2, 2, 2, 2]))
    assert step_inputs.is_open()
    torch.cuda.synchronize()
    assert buf.tolist() == [2, 1, 2, 1, 2, 1, 2, 1]
    assert other.tolist() == [3, 3, 3, 3]

    # The pack reopened: later copies are deferred again.
    step_inputs.copy_(later, _host([5, 6]))
    torch.cuda.synchronize()
    assert later.tolist() == [0, 0]
    step_inputs.flush()
    torch.cuda.synchronize()
    assert later.tolist() == [5, 6]


def test_a_device_copy_with_a_dtype_change_is_not_packed(pack):
    dst = _zeros(3, dtype=torch.int64)
    src = torch.tensor([4, 5, 6], dtype=torch.int32, device="cuda")

    step_inputs.begin()
    step_inputs.copy_(dst, src)
    torch.cuda.synchronize()

    assert dst.tolist() == [4, 5, 6]
    assert step_inputs.is_open()


def test_a_copy_larger_than_the_staging_buffer_falls_back_in_order(pack):
    numel = StepInputPack.CAPACITY // 4
    big = _zeros(numel)
    big_values = torch.arange(numel, dtype=torch.int32).pin_memory()

    step_inputs.begin()
    step_inputs.copy_(big[:4], _host([-1, -1, -1, -1]))
    step_inputs.copy_(big, big_values)
    torch.cuda.synchronize()

    # The staged prefix landed before the full copy, which therefore wins.
    assert torch.equal(big.cpu(), big_values)
    assert step_inputs.is_open()


def test_copies_that_together_overflow_the_staging_buffer_all_land(pack):
    # Each copy is a quarter of the staging buffer, so the fourth one does not fit.
    chunk = StepInputPack.CAPACITY // 16
    dsts = [_zeros(chunk) for _ in range(6)]
    values = [torch.full((chunk,), i + 1, dtype=torch.int32).pin_memory() for i in range(6)]

    step_inputs.begin()
    for dst, value in zip(dsts, values):
        step_inputs.copy_(dst, value)
    step_inputs.flush()
    torch.cuda.synchronize()

    for i, dst in enumerate(dsts):
        assert torch.equal(dst.cpu(), values[i]), f"copy {i} did not land"


def test_more_copies_than_descriptors_all_land(pack):
    count = StepInputPack.MAX_DESC + 40
    dst = _zeros(count)

    step_inputs.begin()
    for i in range(count):
        step_inputs.copy_(dst[i : i + 1], _host([i + 1]))
    step_inputs.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == list(range(1, count + 1))


def test_a_row_strided_copy_that_needs_more_descriptors_than_remain_falls_back(pack):
    rows = 8
    table = torch.full((rows, 4), -1, dtype=torch.int32, device="cuda")
    filler = _zeros(StepInputPack.MAX_DESC - 2)
    rows_values = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)

    step_inputs.begin()
    for i in range(filler.numel()):
        step_inputs.copy_(filler[i : i + 1], _host([i + 1]))
    # Two descriptors remain; the slice needs one per row, so it flushes and copies directly.
    step_inputs.copy_(table[:, :3], rows_values.pin_memory())
    step_inputs.flush()
    torch.cuda.synchronize()

    assert filler.tolist() == list(range(1, filler.numel() + 1))
    assert torch.equal(table[:, :3].cpu(), rows_values)
    assert table[:, 3].tolist() == [-1] * rows


def test_mixed_device_copies_past_the_descriptor_limit_all_land(pack):
    count = StepInputPack.MAX_DESC + 10
    src = torch.arange(1, count + 1, dtype=torch.int32, device="cuda")
    dst = _zeros(count)

    step_inputs.begin()
    for i in range(count):
        step_inputs.copy_(dst[i : i + 1], src[i : i + 1])
    step_inputs.flush()
    torch.cuda.synchronize()

    assert torch.equal(dst, src)


def test_begin_on_an_open_pack_issues_it_first(pack):
    dst = _zeros(2)

    step_inputs.begin()
    step_inputs.copy_(dst, _host([8, 9]))
    step_inputs.begin()
    torch.cuda.synchronize()

    assert dst.tolist() == [8, 9]


def test_reusing_a_pinned_staging_buffer_waits_for_its_upload(pack):
    """With every staging buffer in the ring used once, the next pack reuses the first one.

    Its upload is queued behind a long kernel, so rewriting the buffer without waiting would
    upload the newer bytes in place of the older ones.
    """
    ring = StepInputPack.RING
    dsts = [_zeros(4) for _ in range(ring + 1)]
    values = [[i * 10 + j for j in range(4)] for i in range(ring + 1)]

    step_inputs.begin()
    torch.cuda._sleep(_SLEEP_CYCLES)
    step_inputs.copy_(dsts[0], _host(values[0]))
    step_inputs.flush()
    first_upload = pack.events[0]
    for i in range(1, ring):
        step_inputs.begin()
        step_inputs.copy_(dsts[i], _host(values[i]))
        step_inputs.flush()
    assert not first_upload.query(), "the queued kernel finished too early for this check"

    step_inputs.begin()
    assert first_upload.query()
    step_inputs.copy_(dsts[ring], _host(values[ring]))
    step_inputs.flush()
    torch.cuda.synchronize()

    for i, dst in enumerate(dsts):
        assert dst.tolist() == values[i], f"pack {i} uploaded the wrong bytes"


def test_a_copy_while_capturing_is_not_deferred(pack):
    dst = _zeros(3)
    src = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")

    step_inputs.begin()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step_inputs.copy_(dst, src)
    step_inputs.flush()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0]

    graph.replay()
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


def test_discard_drops_the_staged_copies(pack):
    dst = _zeros(3)

    step_inputs.begin()
    step_inputs.copy_(dst, _host([1, 2, 3]))
    step_inputs.copy_(dst, torch.ones(3, dtype=torch.int32, device="cuda"))
    step_inputs.discard()
    assert not pack.open and pack.staged == [] and pack.h2d == [] and pack.d2d == []
    step_inputs.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [0, 0, 0]


def test_after_a_discarded_step_identical_values_are_sent_again(pack):
    """The runner's recovery: discard the pack and clear the skip cache."""
    skip_cache = H2DSkipCache()
    dst = _zeros(3)
    values = _host([4, 5, 6])

    with h2d_cache.active(skip_cache):
        step_inputs.begin()
        assert copy_h2d_if_changed(dst, values, pack=True) is True
        step_inputs.discard()
        torch.cuda.synchronize()
        assert dst.tolist() == [0, 0, 0]
        # Without clearing, the cache would vouch for values that never landed.
        assert copy_h2d_if_changed(dst, values, pack=True) is False
        skip_cache.clear()

        step_inputs.begin()
        assert copy_h2d_if_changed(dst, values, pack=True) is True
        step_inputs.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [4, 5, 6]


def test_skip_cache_copies_join_the_active_pack(pack):
    skip_cache = H2DSkipCache()
    dst = _zeros(3)
    values = _host([4, 5, 6])

    with h2d_cache.active(skip_cache):
        step_inputs.begin()
        assert copy_h2d_if_changed(dst, values, pack=True) is True
        torch.cuda.synchronize()
        assert dst.tolist() == [0, 0, 0]
        # Already staged with these values: the second copy is skipped.
        assert copy_h2d_if_changed(dst, values, pack=True) is False
        step_inputs.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [4, 5, 6]


# ---------------------------------------------------------------------------
# Isolation between packs
# ---------------------------------------------------------------------------


def test_a_nested_pack_does_not_see_the_outer_pack(pack):
    outer_dst = _zeros(2)
    inner_dst = _zeros(2)
    other = StepInputPack(_device())

    step_inputs.begin()
    step_inputs.copy_(outer_dst, _host([1, 2]))
    with step_inputs.active(other):
        assert not step_inputs.is_open()
        step_inputs.copy_(inner_dst, _host([3, 4]))  # plain copy: the inner pack is closed
        step_inputs.begin()
        step_inputs.copy_(inner_dst, _host([5, 6]))
        torch.cuda.synchronize()
        assert inner_dst.tolist() == [3, 4]
        assert outer_dst.tolist() == [0, 0]
        step_inputs.flush()
    assert step_inputs.is_open() and len(pack.h2d) == 1
    step_inputs.flush()
    torch.cuda.synchronize()

    assert inner_dst.tolist() == [5, 6]
    assert outer_dst.tolist() == [1, 2]


def test_active_none_disables_packing_inside_an_outer_pack(pack):
    dst = _zeros(2)

    step_inputs.begin()
    with step_inputs.active(None):
        assert not step_inputs.is_open()
        step_inputs.copy_(dst, _host([7, 8]))
        torch.cuda.synchronize()
        assert dst.tolist() == [7, 8]
    assert pack.h2d == []


def test_an_open_pack_on_one_thread_is_invisible_to_another(pack):
    mine = _zeros(2)
    theirs = _zeros(2)
    seen = {}

    step_inputs.begin()
    step_inputs.copy_(mine, _host([1, 1]))

    def other_thread():
        seen["open"] = step_inputs.is_open()
        step_inputs.copy_(theirs, _host([2, 2]))
        step_inputs.flush()
        torch.cuda.synchronize()
        seen["theirs"] = theirs.tolist()
        seen["mine"] = mine.tolist()

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()

    assert seen == {"open": False, "theirs": [2, 2], "mine": [0, 0]}
    assert step_inputs.is_open() and len(pack.h2d) == 1
    step_inputs.flush()
    torch.cuda.synchronize()
    assert mine.tolist() == [1, 1]


def test_threads_with_their_own_packs_keep_them_apart():
    packs = [StepInputPack(_device()) for _ in range(2)]
    dsts = [_zeros(4) for _ in range(2)]
    barrier = threading.Barrier(2)
    results = {}

    def worker(index):
        with step_inputs.active(packs[index]):
            step_inputs.begin()
            step_inputs.copy_(dsts[index], _host([index + 1] * 4))
            barrier.wait()
            # Both packs are open with one staged copy each.
            results[index] = (packs[index].open, len(packs[index].h2d))
            barrier.wait()
            step_inputs.flush()
            torch.cuda.synchronize()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {0: (True, 1), 1: (True, 1)}
    assert dsts[0].tolist() == [1] * 4
    assert dsts[1].tolist() == [2] * 4


def _random_pack_run(parallel: bool, seed: int, overlap: bool):
    """Stage a fixed random mix of host copies (2-D page-table rows included) and device copies that read a host
    copy's destination; return every destination after the flush."""
    g = torch.Generator().manual_seed(seed)
    pack = StepInputPack(_device(), parallel_scatter=parallel)
    big = _zeros(4096)
    table = torch.zeros((8, 64), dtype=torch.int32, device=_device())
    out = _zeros(256)
    with step_inputs.active(pack):
        step_inputs.begin()
        for k in range(24):
            n = int(torch.randint(1, 64, (1,), generator=g))
            step_inputs.copy_(
                big[k * 128 : k * 128 + n],
                torch.randint(0, 1 << 30, (n,), generator=g, dtype=torch.int32),
            )
        step_inputs.copy_(
            table[:3, :40], torch.randint(0, 1 << 30, (3, 40), generator=g, dtype=torch.int32)
        )
        if overlap:
            # A later host copy overwrites part of an earlier one: order matters, so the serial scatter must run.
            step_inputs.copy_(big[0:4], torch.full((4,), -7, dtype=torch.int32))
        step_inputs.copy_(out[:64], big[:64])
        step_inputs.copy_(out[64:96], big[128:160])
        step_inputs.flush()
    torch.cuda.synchronize()
    return big.cpu(), table.cpu(), out.cpu()


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint", "overlapping"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parallel_scatter_writes_what_the_serial_scatter_writes(seed, overlap):
    serial = _random_pack_run(False, seed, overlap)
    parallel = _random_pack_run(True, seed, overlap)
    for a, b in zip(serial, parallel):
        assert torch.equal(a, b)


def test_parallel_scatter_only_for_order_free_packs():
    assert step_inputs._parallel_ok([(0, 0, 8), (8, 0, 8)], [])
    assert not step_inputs._parallel_ok([(0, 0, 8), (4, 0, 8)], [])
    # A device copy may read a host copy's destination (the launches run in order) ...
    assert step_inputs._parallel_ok([(0, 0, 8)], [(100, 0, 8)])
    # ... but not another device copy's destination, and device destinations must be disjoint.
    assert not step_inputs._parallel_ok([], [(100, 0, 8), (200, 100, 8)])
    assert not step_inputs._parallel_ok([], [(100, 0, 8), (104, 300, 8)])


def test_parallel_scatter_is_off_by_default():
    assert not StepInputPack(_device()).parallel_scatter


def _parallel_scatter_compiles() -> int:
    """How many variants of the parallel scatter kernel Triton holds compiled (any device)."""
    kernel = step_inputs._parallel_scatter_kernel
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(entry[0]) for entry in caches.values())
    return sum(len(entry) for entry in kernel.cache.values())


def test_parallel_scatter_compiles_nothing_after_the_warm_up():
    # The warm-up compiles both DEVICE_SRC variants; packs with any number of host copies (which moves the device
    # copies' offset `first`) must reuse them, so no serving-step flush compiles.
    pack = StepInputPack(_device(), parallel_scatter=True)
    compiled = _parallel_scatter_compiles()
    dst = _zeros(4096)
    out = _zeros(64)
    with step_inputs.active(pack):
        for n_host in (1, 2, 3, 5, 15, 16, 17, 24):
            step_inputs.begin()
            for k in range(n_host):
                step_inputs.copy_(dst[k * 64 : k * 64 + 4], torch.full((4,), k, dtype=torch.int32))
            step_inputs.copy_(out[:4], dst[:4])
            step_inputs.flush()
    torch.cuda.synchronize()
    assert _parallel_scatter_compiles() == compiled
    assert out[:4].tolist() == [0] * 4
    assert dst[23 * 64 : 23 * 64 + 4].tolist() == [23] * 4


# ---------------------------------------------------------------------------
# Flags off: the GB path, byte for byte
# ---------------------------------------------------------------------------


def _gb_flush(pack):
    """StepInputPack.flush as of the GB base (1af1e3272c), verbatim: the reference the flags-off path must match."""
    if not pack.open:
        return
    pack.open = False
    if not pack.h2d and not pack.d2d:
        return
    host = pack.host[pack.parity]
    desc = [len(pack.h2d), len(pack.d2d)] + [
        value for copy in pack.h2d + pack.d2d for value in copy
    ]
    host[: 8 * len(desc)].view(torch.int64).copy_(torch.tensor(desc, dtype=torch.int64))
    pack.dev[: pack.used].copy_(host[: pack.used], non_blocking=True)
    if pack.parallel_scatter and step_inputs._parallel_ok(pack.h2d, pack.d2d):
        stage_i64 = pack.dev.view(torch.int64)
        if pack.h2d:
            step_inputs._parallel_scatter_kernel[(len(pack.h2d),)](
                pack.dev, stage_i64, 0, DEVICE_SRC=False, BLOCK=256, num_warps=2
            )
        if pack.d2d:
            step_inputs._parallel_scatter_kernel[(len(pack.d2d),)](
                pack.dev, stage_i64, len(pack.h2d), DEVICE_SRC=True, BLOCK=256, num_warps=2
            )
    else:
        torch.ops.trtllm.step_inputs_scatter(pack.dev)
    event = pack.events[pack.parity]
    if event is None:
        event = pack.events[pack.parity] = torch.cuda.Event()
    event.record()
    pack.parity = (pack.parity + 1) % pack.RING
    pack._clear()


def _flags_off_run(parallel: bool, seed: int, overlap: bool, how: str):
    """``_random_pack_run`` issued by ``flush()``, ``flush(in_graph=True)`` or the GB reference; returns every
    destination and the uploaded staging bytes."""
    g = torch.Generator().manual_seed(seed)
    pack = StepInputPack(_device(), parallel_scatter=parallel)
    big = _zeros(4096)
    table = torch.zeros((8, 64), dtype=torch.int32, device=_device())
    out = _zeros(256)
    with step_inputs.active(pack):
        step_inputs.begin()
        for k in range(24):
            n = int(torch.randint(1, 64, (1,), generator=g))
            step_inputs.copy_(
                big[k * 128 : k * 128 + n],
                torch.randint(0, 1 << 30, (n,), generator=g, dtype=torch.int32),
            )
        step_inputs.copy_(
            table[:3, :40], torch.randint(0, 1 << 30, (3, 40), generator=g, dtype=torch.int32)
        )
        if overlap:
            step_inputs.copy_(big[0:4], torch.full((4,), -7, dtype=torch.int32))
        step_inputs.copy_(out[:64], big[:64])
        step_inputs.copy_(out[64:96], big[128:160])
        used = pack.used
        if how == "flush":
            step_inputs.flush()
        elif how == "flush_in_graph":
            step_inputs.flush(in_graph=True)
        else:
            _gb_flush(pack)
    torch.cuda.synchronize()
    assert not pack.open and pack.staged == []
    assert (pack.graph_windows, pack.eager_fallbacks, pack.early_uploads) == (0, 0, 0)
    return big.cpu(), table.cpu(), out.cpu(), _uploaded(pack, used, (big, table, out)), pack.parity


def _uploaded(pack, used, tensors):
    """The staging bytes the flush uploaded, with each device address written as (tensor index, byte offset) so two
    runs compare: the header, every descriptor, and every host copy's staged data bytes."""
    staged = pack.dev[:used].cpu()
    words = staged[: pack.data_start].view(torch.int64).tolist()
    num_h2d, num_d2d = words[0], words[1]
    spans = [(t.data_ptr(), t.numel() * t.element_size()) for t in tensors]

    def where(addr):
        for i, (base, nbytes) in enumerate(spans):
            if base <= addr < base + nbytes:
                return (i, addr - base)
        raise AssertionError(f"address {addr:#x} outside the test tensors")

    copies = []
    for k in range(num_h2d + num_d2d):
        dst, src, nbytes = words[2 + 3 * k : 5 + 3 * k]
        if k < num_h2d:
            copies.append((where(dst), src, nbytes, staged[src : src + nbytes].tolist()))
        else:
            copies.append((where(dst), where(src), nbytes))
    return num_h2d, num_d2d, copies


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint", "overlapping"])
@pytest.mark.parametrize("seed", [0, 1])
def test_flags_off_flush_is_the_gb_flush_byte_for_byte(parallel, overlap, seed):
    """With pack_in_graph off, flush() and flush(in_graph=True) upload the same staging bytes, write the same
    destinations and advance the ring like the GB base's flush."""
    reference = _flags_off_run(parallel, seed, overlap, "gb")
    for how in ("flush", "flush_in_graph"):
        got = _flags_off_run(parallel, seed, overlap, how)
        for a, b in zip(reference[:3], got[:3]):
            assert torch.equal(a, b), how
        assert got[3] == reference[3], how
        assert got[4] == reference[4], how


def test_flags_off_pack_allocates_no_window_and_has_no_prologue():
    for kwargs in ({}, {"parallel_scatter": True}):
        pack = StepInputPack(_device(), **kwargs)
        assert not pack.in_graph and not pack.early_copy
        for name in ("ghost", "ghost_free", "gdev", "copy_stream", "staged_ready", "prologue_done"):
            assert not hasattr(pack, name), name
        with step_inputs.active(pack):
            assert step_inputs.capture_prologue() is None
        # A closed flags-off pack: flush(in_graph=True) is a no-op, like flush().
        pack.flush(in_graph=True)
        pack.quiesce_graph_window()
    assert step_inputs.capture_prologue() is None


def test_a_pack_built_without_init_keeps_every_in_graph_item_off():
    """The class-level defaults (object.__new__ executors in tests never run __init__)."""
    pack = object.__new__(StepInputPack)
    assert not pack.parallel_scatter and not pack.in_graph and not pack.early_copy
    assert (pack.graph_windows, pack.eager_fallbacks, pack.early_uploads) == (0, 0, 0)
    pack.quiesce_graph_window()
    assert not pack.capture_prologue()


# ---------------------------------------------------------------------------
# In-graph prologue (pack_in_graph) and early copy (pack_early_copy)
# ---------------------------------------------------------------------------


def _graph_case(early=False):
    """A pack with the in-graph prologue and a graph that starts with it and then reads the destinations."""
    pack = StepInputPack(_device(), parallel_scatter=True, in_graph=True, early_copy=early)
    a, b, c = _zeros(16), _zeros(16), _zeros(16)
    out = _zeros(48)
    graph = torch.cuda.CUDAGraph()
    with step_inputs.active(pack):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            assert step_inputs.capture_prologue() is pack
            out.copy_(torch.cat([a, b, c]))
    return pack, graph, (a, b, c), out


def _step(pack, graph, dsts, values, overlap=False, flush=True):
    a, b, c = dsts
    with step_inputs.active(pack):
        step_inputs.begin()
        step_inputs.copy_(a, _host(values[0]))
        step_inputs.copy_(b, _host(values[1]))
        if overlap:
            step_inputs.copy_(a[:4], _host([-1, -2, -3, -4]))
        step_inputs.copy_(c, a)  # a device copy reading a host copy's destination
        if flush:
            step_inputs.flush(in_graph=True)
        else:
            step_inputs.discard()
            step_inputs.flush(in_graph=True)
    graph.replay()


def test_in_graph_pack_is_issued_by_the_replay():
    pack, graph, dsts, out = _graph_case()
    for step in range(5):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        _step(pack, graph, dsts, (va, vb))
        torch.cuda.synchronize()
        assert out.tolist() == va + vb + va, step
        # Nothing was issued outside the graph: the window carried the copies.
        assert int(pack.ghost[:8].view(torch.int64)[0]) == 2
    assert (pack.graph_windows, pack.eager_fallbacks, pack.early_uploads) == (5, 0, 0)


def test_in_graph_pack_not_order_free_is_issued_eagerly():
    pack, graph, dsts, out = _graph_case()
    va = list(range(16))
    vb = list(range(100, 116))
    _step(pack, graph, dsts, (va, vb), overlap=True)
    torch.cuda.synchronize()
    expect_a = [-1, -2, -3, -4] + va[4:]
    assert out.tolist() == expect_a + vb + expect_a
    # The window was zeroed, so the prologue wrote nothing over the eager result.
    assert not pack.ghost[:16].any()
    assert (pack.graph_windows, pack.eager_fallbacks) == (0, 1)


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_in_graph_pack_over_the_window_is_issued_eagerly(early):
    pack, graph, dsts, out = _graph_case(early)
    big = _zeros(StepInputPack.GRAPH_WINDOW // 4)
    va = list(range(16))
    vb = list(range(200, 216))
    with step_inputs.active(pack):
        step_inputs.begin()
        step_inputs.copy_(dsts[0], _host(va))
        step_inputs.copy_(dsts[1], _host(vb))
        step_inputs.copy_(big, torch.arange(big.numel(), dtype=torch.int32))
        step_inputs.copy_(dsts[2], dsts[0])
        assert pack.used > StepInputPack.GRAPH_WINDOW
        step_inputs.flush(in_graph=True)
    graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == va + vb + va
    assert big.cpu().tolist() == list(range(big.numel()))
    assert (pack.graph_windows, pack.eager_fallbacks, pack.early_uploads) == (0, 1, 0)


def test_in_graph_replay_without_a_pack_reapplies_nothing():
    pack, graph, dsts, out = _graph_case()
    _step(pack, graph, dsts, (list(range(16)), list(range(16))))
    torch.cuda.synchronize()
    dsts[0].fill_(7)
    _step(pack, graph, dsts, ([0] * 16, [0] * 16), flush=False)
    torch.cuda.synchronize()
    assert dsts[0].tolist() == [7] * 16


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_in_graph_pack_lands_every_step(early):
    pack, graph, dsts, out = _graph_case(early)
    assert pack.early_copy is early
    for step in range(6):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        _step(pack, graph, dsts, (va, vb), overlap=step % 2 == 1)
        torch.cuda.synchronize()
        head = [-1, -2, -3, -4] if step % 2 == 1 else []
        expect_a = head + va[len(head) :]
        assert out.tolist() == expect_a + vb + expect_a, step
    assert (pack.graph_windows, pack.eager_fallbacks) == (3, 3)
    assert pack.early_uploads == (3 if early else 0)


def _history_run(pack, graph, dsts, out, steps, before_replay=None, before_upload=None):
    """Steps back to back without host syncs; each replay's output is copied to a history row behind it (same
    stream), so a later upload that overwrote the window too early shows up as the wrong step's values."""
    hist = _zeros(steps, out.numel())
    expect = []
    for step in range(steps):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        expect.append(va + vb + va)
        a, b, c = dsts
        with step_inputs.active(pack):
            step_inputs.begin()
            step_inputs.copy_(a, _host(va))
            step_inputs.copy_(b, _host(vb))
            step_inputs.copy_(c, a)
            if before_upload is not None:
                before_upload(step)
            step_inputs.flush(in_graph=True)
        if before_replay is not None:
            before_replay(step)
        graph.replay()
        hist[step].copy_(out)
    torch.cuda.synchronize()
    return hist.tolist(), expect


def test_early_copy_waits_for_the_previous_prologue_before_overwriting_the_window():
    """A slow GPU: the replay of step N (and its prologue) is held back behind a long kernel while the host already
    uploads step N+1; that upload must wait for step N's prologue to consume the window."""
    pack, graph, dsts, out = _graph_case(early=True)
    hist, expect = _history_run(
        pack, graph, dsts, out, 4, before_replay=lambda step: torch.cuda._sleep(_SLEEP_CYCLES // 4)
    )
    assert hist == expect


def test_early_copy_prologue_waits_for_a_slow_upload():
    """A slow copy stream: each step's upload is held back behind a long kernel on the copy stream; the prologue must
    wait for it (staged_ready), and the next host fill must wait for the upload to have read the window."""
    pack, graph, dsts, out = _graph_case(early=True)

    def slow_copy(step):
        with torch.cuda.stream(pack.copy_stream):
            torch.cuda._sleep(_SLEEP_CYCLES // 4)

    hist, expect = _history_run(pack, graph, dsts, out, 4, before_upload=slow_copy)
    assert hist == expect


def test_early_copy_with_a_slow_host():
    """A slow host: the GPU idles between steps (the window lands long before the replay)."""
    import time

    pack, graph, dsts, out = _graph_case(early=True)
    hist, expect = _history_run(
        pack, graph, dsts, out, 3, before_replay=lambda step: time.sleep(0.05)
    )
    assert hist == expect


def test_early_copy_snapshots_the_host_bytes_before_a_stalled_upload():
    """PR-CTX (d)(1): stall the copy stream, issue the early copy, overwrite the host source, then sync: the device
    holds the pre-overwrite values. The next step's host fill also runs while the first upload is still stalled (it
    must wait for that upload to have read the pinned window before it overwrites it)."""
    pack, graph, dsts, out = _graph_case(early=True)
    a, b, c = dsts
    hist = _zeros(2, out.numel())
    va, vb = list(range(16)), list(range(500, 516))
    src_a, src_b = _host(va), _host(vb)
    with torch.cuda.stream(pack.copy_stream):
        torch.cuda._sleep(1 << 30)
    with step_inputs.active(pack):
        step_inputs.begin()
        step_inputs.copy_(a, src_a)
        step_inputs.copy_(b, src_b)
        step_inputs.copy_(c, a)
        step_inputs.flush(in_graph=True)
        # The upload is queued behind the stall: it has not read the window yet.
        assert not pack.staged_ready.query()
        src_a.fill_(-9)
        src_b.fill_(-9)
    graph.replay()
    hist[0].copy_(out)
    # Step 2's fill blocks on ghost_free (the stalled upload) before it writes the pinned window.
    va2, vb2 = [7] * 16, [8] * 16
    with step_inputs.active(pack):
        step_inputs.begin()
        step_inputs.copy_(a, _host(va2))
        step_inputs.copy_(b, _host(vb2))
        step_inputs.copy_(c, a)
        step_inputs.flush(in_graph=True)
    graph.replay()
    hist[1].copy_(out)
    torch.cuda.synchronize()
    assert hist[0].tolist() == va + vb + va
    assert hist[1].tolist() == va2 + vb2 + va2
    assert (pack.graph_windows, pack.early_uploads) == (2, 2)


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_in_graph_replay_after_quiesce_reapplies_nothing(early):
    pack, graph, dsts, out = _graph_case(early)
    _step(pack, graph, dsts, (list(range(16)), list(range(16))))
    torch.cuda.synchronize()
    dsts[0].fill_(7)
    pack.quiesce_graph_window()
    graph.replay()
    torch.cuda.synchronize()
    assert dsts[0].tolist() == [7] * 16


def test_early_copy_needs_in_graph():
    assert not StepInputPack(_device(), parallel_scatter=True, early_copy=True).early_copy


def test_in_graph_needs_parallel_scatter_and_is_off_by_default():
    assert not StepInputPack(_device()).in_graph
    assert not StepInputPack(_device(), in_graph=True).in_graph


def test_in_graph_replay_outside_the_pack_context_reapplies_nothing():
    pack, graph, dsts, out = _graph_case()
    _step(pack, graph, dsts, (list(range(16)), list(range(16))))
    torch.cuda.synchronize()
    dsts[0].fill_(7)
    # What the graph runner does for a replay outside the pack's context.
    assert not step_inputs.is_active(pack)
    pack.quiesce_graph_window()
    graph.replay()
    torch.cuda.synchronize()
    assert dsts[0].tolist() == [7] * 16


def _kernel_compiles(kernel) -> int:
    """How many variants of a Triton kernel are compiled (any device)."""
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(entry[0]) for entry in caches.values())
    return sum(len(entry) for entry in kernel.cache.values())


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_in_graph_kernels_compile_nothing_after_the_warm_up(early):
    """The pack's warm-up compiles both scatter kernels' variants; captures and steps with any number of copies (in
    the window, over it, not order-free, quiesced) compile nothing more."""
    pack = StepInputPack(_device(), parallel_scatter=True, in_graph=True, early_copy=early)
    kernels = (step_inputs._parallel_scatter_kernel, step_inputs._graph_scatter_kernel)
    compiled = [_kernel_compiles(kernel) for kernel in kernels]
    dst = _zeros(8192)
    out = _zeros(64)
    graph = torch.cuda.CUDAGraph()
    with step_inputs.active(pack):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            assert step_inputs.capture_prologue() is pack
            out.add_(1)
    for n_host in (1, 2, 3, 5, 15, 16, 17, 24, 120):
        with step_inputs.active(pack):
            step_inputs.begin()
            for k in range(n_host):
                step_inputs.copy_(dst[k * 64 : k * 64 + 4], torch.full((4,), k, dtype=torch.int32))
            for k in range(n_host % 7):
                step_inputs.copy_(out[k * 4 : k * 4 + 4], dst[k * 64 : k * 64 + 4])
            if n_host == 17:
                step_inputs.copy_(
                    dst[4096:], torch.ones(4096, dtype=torch.int32)
                )  # over the window
            step_inputs.flush(in_graph=True)
        graph.replay()
        if n_host == 5:
            pack.quiesce_graph_window()
            graph.replay()
    torch.cuda.synchronize()
    assert [_kernel_compiles(kernel) for kernel in kernels] == compiled
    assert dst[119 * 64 : 119 * 64 + 4].tolist() == [119] * 4
    assert pack.eager_fallbacks == 1 and pack.graph_windows == 8


# ---------------------------------------------------------------------------
# pack_in_graph through the real CUDAGraphRunner capture -> replay
# ---------------------------------------------------------------------------


def _runner_case(early, in_graph=True):
    from _torch.helpers import create_mock_cuda_graph_runner

    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType

    batch_size = 4
    runner = create_mock_cuda_graph_runner(batch_size, max_num_tokens=128)
    key = KeyType(batch_size=batch_size, draft_len=0)
    num_tokens = runner._get_num_tokens_for_key(key)
    pack = StepInputPack(_device(), parallel_scatter=True, in_graph=in_graph, early_copy=early)
    meta = _zeros(8)
    out = torch.zeros(2 * num_tokens + meta.numel(), dtype=torch.int64, device="cuda")
    attn_metadata = object()

    def forward_fn(inputs):
        out.copy_(
            torch.cat(
                [
                    inputs["input_ids"].to(torch.int64),
                    inputs["position_ids"].flatten().to(torch.int64),
                    meta.to(torch.int64),
                ]
            )
        )
        return out

    def inputs(value):
        return {
            "attn_metadata": attn_metadata,
            "input_ids": torch.full((num_tokens,), value, dtype=torch.int32, device="cuda"),
            "position_ids": torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, -1)
            + value,
        }

    with step_inputs.active(pack):
        runner.capture(key, forward_fn, inputs(0))
    return runner, key, pack, meta, out, inputs, num_tokens


def _runner_step(runner, key, pack, meta, inputs, value):
    with step_inputs.active(pack):
        step_inputs.begin()
        # A metadata upload the engine stages before the replay (as the MSA metadata does).
        step_inputs.copy_(meta, _host([value * 10 + i for i in range(meta.numel())]))
        runner.replay(key, inputs(value))


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_in_graph_pack_through_the_graph_runner_capture_and_replay(early):
    """capture() adds the active pack's prologue to the graph; replay() with fresh inputs only fills the window (the
    static input copies and the metadata upload are issued by the replay's prologue), every step lands."""
    runner, key, pack, meta, out, inputs, n = _runner_case(early)
    assert runner._step_input_prologue_keys == {key: pack}
    for value in range(1, 6):
        _runner_step(runner, key, pack, meta, inputs, value)
        torch.cuda.synchronize()
        expect = [value] * n + [value + i for i in range(n)] + [value * 10 + i for i in range(8)]
        assert out.tolist() == expect, value
        assert pack.graph_windows == value and pack.eager_fallbacks == 0
    assert pack.early_uploads == (5 if early else 0)
    # Replayed outside the pack's context: the inputs are plain copies and the prologue re-issues nothing.
    meta.fill_(-3)
    runner.replay(key, inputs(9))
    torch.cuda.synchronize()
    assert out.tolist() == [9] * n + [9 + i for i in range(n)] + [-3] * 8
    runner.clear()
    assert runner._step_input_prologue_keys is None


def test_graph_runner_flags_off_replays_through_the_stock_flush(monkeypatch):
    """A pack without in_graph: the capture adds no prologue and replay() issues the pack with flush() exactly as on
    the GB base (no in_graph argument), before the graph replay."""
    runner, key, pack, meta, out, inputs, n = _runner_case(False, in_graph=False)
    assert runner._step_input_prologue_keys is None
    calls = []
    real_flush = step_inputs.flush

    def recording_flush(*args, **kwargs):
        calls.append((args, kwargs))
        return real_flush(*args, **kwargs)

    monkeypatch.setattr(step_inputs, "flush", recording_flush)
    _runner_step(runner, key, pack, meta, inputs, 3)
    torch.cuda.synchronize()
    assert calls == [((), {})]
    assert out.tolist() == [3] * n + [3 + i for i in range(n)] + [30 + i for i in range(8)]
    assert (pack.graph_windows, pack.eager_fallbacks) == (0, 0)
