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
