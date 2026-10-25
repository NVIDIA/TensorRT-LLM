# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A step's copies routed by a copy plan: deferred into one upload and one scatter launch, or skipped when unchanged."""

import threading

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.device_copy import (
    CHECK_ENV,
    CopyPlan,
    CopySite,
    CopySiteEnum,
    DeviceCopier,
    Direction,
    _Pack,
    copy_plan_of,
)


def _scatter_op_registered() -> bool:
    return torch.cuda.is_available() and hasattr(torch.ops.trtllm, "device_copy_scatter")


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not _scatter_op_registered(),
        reason="torch.ops.trtllm.device_copy_scatter is not registered (built from "
        "cpp/tensorrt_llm/thop/deviceCopyScatterOp.cpp)",
    ),
]

# About half a second of GPU time: long enough that work queued behind it is still pending when
# the host checks, short enough for a unit test.
_SLEEP_CYCLES = 1_000_000_000


class _Site(CopySiteEnum):
    H2D = ("test.h2d", Direction.H2D)
    D2D = ("test.d2d", Direction.D2D)
    # Deferred and skipped when unchanged.
    SKIP = ("test.skip", Direction.H2D)
    TAIL = ("test.tail", Direction.FILL)
    # In neither set of the plan.
    PLAIN = ("test.plain", Direction.H2D)
    # Skipped only.
    SKIP_ONLY = ("test.skip_only", Direction.H2D)


_PLAN = CopyPlan(
    defer={_Site.H2D, _Site.D2D, _Site.SKIP},
    skip_unchanged={_Site.SKIP, _Site.TAIL, _Site.SKIP_ONLY},
)


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _new_copier(plan: CopyPlan = _PLAN) -> DeviceCopier:
    return DeviceCopier(plan, _device())


@pytest.fixture
def copier():
    """A fresh copier, active for the test body."""
    active = _new_copier()
    with device_copy.active(active):
        yield active
        device_copy.flush()
    torch.cuda.synchronize()


def _copy(dst, src) -> bool:
    """A copy through the deferred site of ``src``'s direction."""
    return device_copy.copy(_Site.H2D if src.device.type == "cpu" else _Site.D2D, dst, src)


def _skip_copy(dst, src) -> bool:
    return device_copy.copy(_Site.SKIP, dst, src)


def _host(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype).pin_memory()


def _zeros(*shape, dtype=torch.int32):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_the_plan_routes_each_site():
    plan = CopyPlan(defer={_Site.H2D}, skip_unchanged={_Site.SKIP_ONLY})
    deferred, skipped, plain = (_zeros(2) for _ in range(3))
    values = _host([4, 5])

    with device_copy.active(_new_copier(plan)):
        device_copy.begin()
        assert device_copy.copy(_Site.H2D, deferred, values) is True
        # Skipped sites are not deferred unless the plan says so; the repeat is skipped.
        assert [device_copy.copy(_Site.SKIP_ONLY, skipped, values) for _ in range(2)] == [
            True,
            False,
        ]
        # A site in neither set is a plain copy, every time.
        assert [device_copy.copy(_Site.PLAIN, plain, values) for _ in range(2)] == [True, True]
        torch.cuda.synchronize()
        assert (deferred.tolist(), skipped.tolist(), plain.tolist()) == ([0, 0], [4, 5], [4, 5])
        with pytest.raises(ValueError, match=r"_Site\.H2D is declared h2d"):
            device_copy.copy(_Site.H2D, deferred, torch.ones(2, dtype=torch.int32, device="cuda"))
        device_copy.flush()
    torch.cuda.synchronize()
    assert deferred.tolist() == [4, 5]

    with pytest.raises(ValueError, match="cannot be deferred"):
        CopyPlan(defer={_Site.TAIL})
    with pytest.raises(ValueError, match="cannot be skipped"):
        CopyPlan(skip_unchanged={CopySite.GRAPH_INPUT_IDS})
    with pytest.raises(TypeError, match="CopySiteEnum"):
        CopyPlan(defer={"attn.kv_lens"})
    model = torch.nn.Module()
    assert copy_plan_of(model) == CopyPlan() and not copy_plan_of(model)
    model.copy_plan = plan
    assert copy_plan_of(model) is plan
    model.copy_plan = {_Site.H2D}
    with pytest.raises(TypeError, match=r"Module\.copy_plan must be a CopyPlan"):
        copy_plan_of(model)


@pytest.mark.parametrize(
    "broken", ["deferred_dst_written", "d2d_src_changed", "skipped_dst_written", "tail_written"]
)
def test_check_mode_names_the_site_whose_assertion_broke(monkeypatch, broken):
    monkeypatch.setenv(CHECK_ENV, "1")
    checked = _new_copier()
    dst = _zeros(4)
    src = torch.arange(4, dtype=torch.int32, device="cuda")

    with device_copy.active(checked):
        # The legitimate pattern passes: a device copy reading what a host copy of the same pack writes.
        staged, static = _zeros(4), _zeros(4)
        device_copy.begin()
        _copy(staged, _host([1, 2, 3, 4]))
        _copy(static, staged)
        device_copy.flush()
        torch.cuda.synchronize()
        assert static.tolist() == [1, 2, 3, 4]

        if broken == "deferred_dst_written":
            device_copy.begin()
            _copy(dst, _host([1, 1, 1, 1]))
            dst[0] = 7
            site, call = r"_Site\.H2D", device_copy.flush
        elif broken == "d2d_src_changed":
            device_copy.begin()
            _copy(dst, src)
            src.add_(1)
            site, call = r"_Site\.D2D", device_copy.flush
        elif broken == "skipped_dst_written":
            device_copy.copy(_Site.SKIP_ONLY, dst, _host([3, 3, 3, 3]))
            dst[1] = 7
            site, call = (
                r"_Site\.SKIP_ONLY",
                lambda: device_copy.copy(_Site.SKIP_ONLY, dst, _host([3, 3, 3, 3])),
            )
        else:
            device_copy.fill_tail(_Site.TAIL, dst, 1, -1)
            dst[2] = 7
            site, call = r"_Site\.TAIL", lambda: device_copy.fill_tail(_Site.TAIL, dst, 1, -1)
        with pytest.raises(RuntimeError, match=f"{CHECK_ENV}: .*{site}"):
            call()
        device_copy.discard()


# ---------------------------------------------------------------------------
# No active copier
# ---------------------------------------------------------------------------


def test_without_an_active_copier_every_call_is_a_plain_copy():
    dst = _zeros(4)
    d2d_dst = _zeros(4)
    src_dev = torch.arange(4, dtype=torch.int32, device="cuda")
    tail = _zeros(6)

    device_copy.begin()
    assert not device_copy.is_open()
    for _ in range(2):
        assert _copy(dst, _host([1, 2, 3, 4])) is True
        assert _skip_copy(dst, _host([1, 2, 3, 4])) is True
    _copy(d2d_dst, src_dev)
    device_copy.fill_tail(_Site.TAIL, tail, 2, -1)
    tail[3] = 5
    device_copy.fill_tail(_Site.TAIL, tail, 2, -1)
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4]
    assert d2d_dst.tolist() == [0, 1, 2, 3]
    assert tail.tolist() == [0, 0, -1, -1, -1, -1]
    assert device_copy.inputs_unchanged(_Site.SKIP, dst, (1,), _host([1])) is False
    device_copy.flush()
    device_copy.discard()


def test_an_inactive_copier_is_not_used():
    idle = _new_copier()
    dst = _zeros(2)

    _copy(dst, _host([3, 4]))
    torch.cuda.synchronize()

    assert dst.tolist() == [3, 4]
    assert not idle.is_open and idle._pack.staged == []


# ---------------------------------------------------------------------------
# Deferral and ordering
# ---------------------------------------------------------------------------


def test_deferred_host_copy_lands_only_at_flush(copier):
    dst = _zeros(4)
    src = _host([1, 2, 3, 4])

    device_copy.begin()
    assert device_copy.is_open() and copier.is_open
    assert _copy(dst, src) is True
    # The pack snapshots the host bytes at the copy site.
    src.fill_(7)
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0, 0]

    device_copy.flush()
    assert not device_copy.is_open()
    assert copier._pack.staged == []
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4]


def test_deferred_host_copy_into_a_row_strided_2d_slice(copier):
    table = torch.full((4, 6), -1, dtype=torch.int32, device="cuda")
    dst = table[1:3, 2:5]
    assert not dst.is_contiguous() and dst.stride(1) == 1
    expected = table.clone()
    expected[1:3, 2:5] = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    device_copy.begin()
    _copy(dst, _host([[1, 2, 3], [4, 5, 6]]))
    torch.cuda.synchronize()
    assert torch.equal(table, torch.full_like(table, -1))

    device_copy.flush()
    torch.cuda.synchronize()
    assert torch.equal(table, expected)


def test_deferred_host_copy_converts_dtype_on_the_host(copier):
    dst = _zeros(3)

    device_copy.begin()
    _copy(dst, _host([7, -8, 9], dtype=torch.int64))
    device_copy.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [7, -8, 9]


def test_deferred_device_copy_lands_only_at_flush(copier):
    dst = _zeros(5)
    src = torch.arange(5, dtype=torch.int32, device="cuda")

    device_copy.begin()
    _copy(dst, src)
    torch.cuda.synchronize()
    assert dst.tolist() == [0] * 5

    device_copy.flush()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 1, 2, 3, 4]


def test_device_copy_reads_what_a_host_copy_of_the_same_pack_wrote(copier):
    """The graph runner copies the engine's freshly uploaded buffers into its static inputs."""
    staged = _zeros(4)
    static = _zeros(1, 8)

    device_copy.begin()
    _copy(staged, _host([10, 11, 12, 13]))
    _copy(static[:, :4], staged.unsqueeze(0))
    device_copy.flush()
    torch.cuda.synchronize()

    assert static.tolist() == [[10, 11, 12, 13, 0, 0, 0, 0]]


# The scatter's CTA copies descriptor i with thread t on bytes t, t + 256, ... of it. When the
# second copy starts TAIL bytes before the end of the first (TAIL not a multiple of the warp
# size), its first bytes are the ones the first copy writes last, from other warps: without a
# barrier between descriptors, a warp that finished the first copy early writes (or reads) them
# before a slower warp has written its part of the first copy.
_TAILS = [239, 1000, 4073]


@pytest.mark.parametrize("tail", _TAILS)
@pytest.mark.parametrize("kind", ["host", "device", "device_then_host"])
def test_overlapping_destinations_land_in_call_order(copier, kind, tail):
    """The later copy wins on the overlap, as it would eagerly.

    ``device_then_host``: the scatter writes host copies before device copies, so a host copy into bytes a staged
    device copy writes flushes the pack first.
    """
    # Host copies share the 64 KiB staging buffer; device copies are bounded only by memory, and a
    # long first copy lets the warps drift apart.
    n = 24 * 1024 if kind == "host" else 1 << 20
    m = 8 * 1024
    first = torch.full((n,), 1, dtype=torch.uint8)
    second = torch.full((m,), 2, dtype=torch.uint8)
    first = first.pin_memory() if kind == "host" else first.cuda()
    second = second.cuda() if kind == "device" else second.pin_memory()
    start = n - tail
    expected = torch.ones(start + m, dtype=torch.uint8)
    expected[start:] = 2

    mismatches = []
    for trial in range(50):
        buf = torch.zeros(start + m, dtype=torch.uint8, device="cuda")
        device_copy.begin()
        _copy(buf[:n], first)
        _copy(buf[start:], second)
        device_copy.flush()
        torch.cuda.synchronize()
        wrong = int((buf.cpu() != expected).sum())
        if wrong:
            mismatches.append((trial, wrong))

    assert not mismatches, f"(trial, overlap bytes where the earlier copy won): {mismatches}"


@pytest.mark.parametrize("tail", _TAILS)
def test_a_device_copy_reads_the_destination_of_an_earlier_device_copy(copier, tail):
    """A chain of device copies: the second reads the bytes the first writes last."""
    n = 1 << 20
    source = torch.randint(1, 256, (n,), dtype=torch.uint8, device="cuda")
    expected = source[n - tail :].clone()

    mismatches = []
    for trial in range(50):
        middle = torch.zeros(n, dtype=torch.uint8, device="cuda")
        out = torch.zeros(tail, dtype=torch.uint8, device="cuda")
        device_copy.begin()
        _copy(middle, source)
        _copy(out, middle[n - tail :])
        device_copy.flush()
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


def test_a_temporary_device_source_outlives_its_copy(copier):
    """The pack holds the source until the scatter is queued, so its memory is not reused."""
    numel = 1 << 16
    expected = torch.arange(numel, dtype=torch.int32) * 3 + 1
    dst = _zeros(numel)

    device_copy.begin()
    # The only reference to this source is the one the pack keeps.
    _copy(dst, torch.arange(numel, dtype=torch.int32, device="cuda") * 3 + 1)
    assert len(copier._pack.staged) == 2
    # Churn the allocator with same-sized blocks: a freed source block would be handed out here.
    garbage = [torch.full((numel,), -7, dtype=torch.int32, device="cuda") for _ in range(16)]
    torch.cuda.synchronize()
    device_copy.flush()
    del garbage
    # Blocks allocated after the flush are ordered behind the scatter on this stream.
    after = [torch.full((numel,), -9, dtype=torch.int32, device="cuda") for _ in range(16)]
    torch.cuda.synchronize()

    assert torch.equal(dst.cpu(), expected)
    del after


def test_a_temporary_destination_view_outlives_its_copy(copier):
    owner = _zeros(8)

    device_copy.begin()
    _copy(owner.view(2, 4)[1], _host([5, 6, 7, 8]))
    assert len(copier._pack.staged) == 1
    device_copy.flush()
    torch.cuda.synchronize()

    assert owner.tolist() == [0, 0, 0, 0, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def test_an_unpackable_copy_flushes_first_and_keeps_order(copier):
    buf = _zeros(8)
    other = _zeros(4)
    later = _zeros(2)

    device_copy.begin()
    _copy(buf, _host([1] * 8))
    _copy(other, _host([3, 3, 3, 3]))
    # A strided 1-D destination cannot be packed: the pack lands first, then this copy.
    _copy(buf[::2], _host([2, 2, 2, 2]))
    assert device_copy.is_open()
    torch.cuda.synchronize()
    assert buf.tolist() == [2, 1, 2, 1, 2, 1, 2, 1]
    assert other.tolist() == [3, 3, 3, 3]

    # The pack reopened: later copies are deferred again.
    _copy(later, _host([5, 6]))
    torch.cuda.synchronize()
    assert later.tolist() == [0, 0]
    device_copy.flush()
    torch.cuda.synchronize()
    assert later.tolist() == [5, 6]


def test_a_device_copy_with_a_dtype_change_is_not_deferred(copier):
    dst = _zeros(3, dtype=torch.int64)
    src = torch.tensor([4, 5, 6], dtype=torch.int32, device="cuda")

    device_copy.begin()
    _copy(dst, src)
    torch.cuda.synchronize()

    assert dst.tolist() == [4, 5, 6]
    assert device_copy.is_open()


def test_a_copy_larger_than_the_staging_buffer_falls_back_in_order(copier):
    numel = _Pack.CAPACITY // 4
    big = _zeros(numel)
    big_values = torch.arange(numel, dtype=torch.int32).pin_memory()

    device_copy.begin()
    _copy(big[:4], _host([-1, -1, -1, -1]))
    _copy(big, big_values)
    torch.cuda.synchronize()

    # The staged prefix landed before the full copy, which therefore wins.
    assert torch.equal(big.cpu(), big_values)
    assert device_copy.is_open()


def test_copies_that_together_overflow_the_staging_buffer_all_land(copier):
    # Each copy is a quarter of the staging buffer, so the fourth one does not fit.
    chunk = _Pack.CAPACITY // 16
    dsts = [_zeros(chunk) for _ in range(6)]
    values = [torch.full((chunk,), i + 1, dtype=torch.int32).pin_memory() for i in range(6)]

    device_copy.begin()
    for dst, value in zip(dsts, values):
        _copy(dst, value)
    device_copy.flush()
    torch.cuda.synchronize()

    for i, dst in enumerate(dsts):
        assert torch.equal(dst.cpu(), values[i]), f"copy {i} did not land"


def test_more_copies_than_descriptors_all_land(copier):
    count = _Pack.MAX_DESC + 40
    dst = _zeros(count)

    device_copy.begin()
    for i in range(count):
        _copy(dst[i : i + 1], _host([i + 1]))
    device_copy.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == list(range(1, count + 1))


def test_a_row_strided_copy_that_needs_more_descriptors_than_remain_falls_back(copier):
    rows = 8
    table = torch.full((rows, 4), -1, dtype=torch.int32, device="cuda")
    filler = _zeros(_Pack.MAX_DESC - 2)
    rows_values = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)

    device_copy.begin()
    for i in range(filler.numel()):
        _copy(filler[i : i + 1], _host([i + 1]))
    # Two descriptors remain; the slice needs one per row, so it flushes and copies directly.
    _copy(table[:, :3], rows_values.pin_memory())
    device_copy.flush()
    torch.cuda.synchronize()

    assert filler.tolist() == list(range(1, filler.numel() + 1))
    assert torch.equal(table[:, :3].cpu(), rows_values)
    assert table[:, 3].tolist() == [-1] * rows


def test_mixed_device_copies_past_the_descriptor_limit_all_land(copier):
    count = _Pack.MAX_DESC + 10
    src = torch.arange(1, count + 1, dtype=torch.int32, device="cuda")
    dst = _zeros(count)

    device_copy.begin()
    for i in range(count):
        _copy(dst[i : i + 1], src[i : i + 1])
    device_copy.flush()
    torch.cuda.synchronize()

    assert torch.equal(dst, src)


def test_begin_on_an_open_pack_issues_it_first(copier):
    dst = _zeros(2)

    device_copy.begin()
    _copy(dst, _host([8, 9]))
    device_copy.begin()
    torch.cuda.synchronize()

    assert dst.tolist() == [8, 9]


def test_reusing_a_pinned_staging_buffer_waits_for_its_upload(copier):
    """With every staging buffer in the ring used once, the next pack reuses the first one.

    Its upload is queued behind a long kernel, so rewriting the buffer without waiting would
    upload the newer bytes in place of the older ones.
    """
    ring = _Pack.RING
    dsts = [_zeros(4) for _ in range(ring + 1)]
    values = [[i * 10 + j for j in range(4)] for i in range(ring + 1)]

    device_copy.begin()
    torch.cuda._sleep(_SLEEP_CYCLES)
    _copy(dsts[0], _host(values[0]))
    device_copy.flush()
    first_upload = copier._pack.events[0]
    for i in range(1, ring):
        device_copy.begin()
        _copy(dsts[i], _host(values[i]))
        device_copy.flush()
    assert not first_upload.query(), "the queued kernel finished too early for this check"

    device_copy.begin()
    assert first_upload.query()
    _copy(dsts[ring], _host(values[ring]))
    device_copy.flush()
    torch.cuda.synchronize()

    for i, dst in enumerate(dsts):
        assert dst.tolist() == values[i], f"pack {i} uploaded the wrong bytes"


def test_a_copy_while_capturing_is_not_deferred(copier):
    dst = _zeros(3)
    src = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")

    device_copy.begin()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _copy(dst, src)
    device_copy.flush()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0]

    graph.replay()
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


def test_discard_drops_the_staged_copies(copier):
    dst = _zeros(3)

    device_copy.begin()
    _copy(dst, _host([1, 2, 3]))
    _copy(dst, torch.ones(3, dtype=torch.int32, device="cuda"))
    device_copy.discard()
    pack = copier._pack
    assert not pack.open and pack.staged == [] and pack.h2d == [] and pack.d2d == []
    device_copy.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [0, 0, 0]


def test_after_a_discarded_step_identical_values_are_sent_again(copier):
    """The runner's recovery: discard drops the pack and the shadows recorded for it."""
    dst = _zeros(3)
    values = _host([4, 5, 6])

    device_copy.begin()
    assert _skip_copy(dst, values) is True
    device_copy.discard()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0]

    device_copy.begin()
    assert _skip_copy(dst, values) is True
    device_copy.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [4, 5, 6]


def test_skipped_copies_join_the_open_pack(copier):
    dst = _zeros(3)
    values = _host([4, 5, 6])

    device_copy.begin()
    assert _skip_copy(dst, values) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 0, 0]
    # Already staged with these values: the second copy is skipped.
    assert _skip_copy(dst, values) is False
    device_copy.flush()
    torch.cuda.synchronize()

    assert dst.tolist() == [4, 5, 6]


# ---------------------------------------------------------------------------
# Skipping unchanged copies
# ---------------------------------------------------------------------------


def test_a_repeated_identical_copy_is_skipped(copier):
    dst = _zeros(4)

    assert _skip_copy(dst, _host([5, 6, 7, 8])) is True
    assert _skip_copy(dst, _host([5, 6, 7, 8])) is False
    torch.cuda.synchronize()
    assert dst.tolist() == [5, 6, 7, 8]


def test_changed_values_are_copied(copier):
    dst = _zeros(4)

    assert _skip_copy(dst, _host([1, 2, 3, 4])) is True
    assert _skip_copy(dst, _host([1, 2, 3, 5])) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 5]
    assert _skip_copy(dst, _host([1, 2, 3, 5])) is False


def test_the_shadow_is_a_snapshot_of_the_source(copier):
    """Mutating the host source after the copy must not make a later copy look unchanged."""
    dst = _zeros(3)
    src = _host([1, 2, 3])

    assert _skip_copy(dst, src) is True
    torch.cuda.synchronize()
    src[0] = 9
    assert _skip_copy(dst, src) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [9, 2, 3]


def test_a_source_dtype_change_copies(copier):
    dst = _zeros(3)

    assert _skip_copy(dst, _host([1, 2, 3], torch.int32)) is True
    assert _skip_copy(dst, _host([1, 2, 3], torch.int64)) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3]


def test_a_copy_through_another_view_replaces_the_shadow(copier):
    buf = _zeros(8)
    head, tail = buf[:4], buf[4:]
    values = _host([1, 2, 3, 4])

    assert _skip_copy(head, values) is True
    assert _skip_copy(tail, _host([5, 6, 7, 8])) is True
    # The shadow now describes the tail, so the head copy must not be skipped.
    assert _skip_copy(head, values) is True
    assert _skip_copy(head, values) is False
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_an_overlapping_view_never_skips_on_stale_state(copier):
    buf = _zeros(6)
    values = _host([1, 1, 1, 1])

    assert _skip_copy(buf[:4], values) is True
    assert _skip_copy(buf[2:6], _host([2, 2, 2, 2])) is True
    # buf[2:4] was overwritten through the other view: same values, same view, must copy.
    assert _skip_copy(buf[:4], values) is True
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 1, 1, 1, 2, 2]


def test_a_view_with_another_shape_of_the_same_bytes_copies(copier):
    buf = _zeros(4)

    assert _skip_copy(buf, _host([1, 2, 3, 4])) is True
    assert _skip_copy(buf.view(2, 2), _host([[1, 2], [3, 4]])) is True
    assert _skip_copy(buf.view(2, 2), _host([[1, 2], [3, 4]])) is False


def test_a_copy_from_another_site_drops_the_shadow(copier):
    dst = _zeros(4)
    values = _host([1, 2, 3, 4])

    assert _skip_copy(dst, values) is True
    # A device copy and a plain-site copy into the same buffer: the shadow no longer describes it.
    _copy(dst, torch.arange(4, dtype=torch.int32, device="cuda"))
    assert _skip_copy(dst, values) is True
    device_copy.copy(_Site.PLAIN, dst, _host([0, 0, 0, 0]))
    assert _skip_copy(dst, values) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4]


def test_a_host_destination_always_copies(copier):
    dst = torch.zeros(3, dtype=torch.int32)

    assert device_copy.copy(_Site.SKIP_ONLY, dst, _host([1, 2, 3])) is True
    assert device_copy.copy(_Site.SKIP_ONLY, dst, _host([1, 2, 3])) is True
    assert dst.tolist() == [1, 2, 3]


def test_capture_always_copies_and_drops_the_shadow(copier):
    dst = _zeros(4)
    values = _host([4, 3, 2, 1])
    assert _skip_copy(dst, values) is True
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    results = []
    with torch.cuda.graph(graph):
        results.append(_skip_copy(dst, values))
    assert results == [True]

    dst.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert dst.tolist() == [4, 3, 2, 1]
    # The captured copy reruns on every replay, so the shadow cannot vouch for the buffer.
    assert _skip_copy(dst, values) is True


def test_a_repeated_tail_fill_is_skipped_and_a_moved_start_refills(copier):
    buf = _zeros(6)
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    # Writes nothing else is allowed to make; they show which fills ran.
    buf[3] = 5
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    assert buf[3].item() == 5
    device_copy.fill_tail(_Site.TAIL, buf, 4, -1)
    buf[3] = 7
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    assert buf.tolist() == [0, 0, -1, -1, -1, -1]


def test_a_whole_buffer_write_drops_the_tail_record(copier):
    buf = _zeros(6)
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    buf.copy_(torch.arange(6, dtype=torch.int32, device="cuda"))
    device_copy.fill_tail(_Site.TAIL, buf, 6, -1)
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    assert buf.tolist() == [0, 1, -1, -1, -1, -1]


def test_discard_drops_the_tail_records(copier):
    buf = _zeros(6)
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    buf[4] = 9
    device_copy.discard()
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    assert buf[2:].eq(-1).all()


def test_a_tail_record_keeps_its_storage_so_a_new_buffer_cannot_inherit_it(copier):
    buf = _zeros(1 << 20)
    device_copy.fill_tail(_Site.TAIL, buf, 3, -1)
    address = buf.data_ptr()
    del buf
    # Without the record holding the storage, the allocator hands this same block back, and the fill below would be
    # skipped on the freed buffer's record.
    fresh = _zeros(1 << 20)
    assert fresh.data_ptr() != address
    device_copy.fill_tail(_Site.TAIL, fresh, 3, -1)
    assert fresh[3:].eq(-1).all()


# ---------------------------------------------------------------------------
# Isolation between copiers
# ---------------------------------------------------------------------------


def test_a_nested_copier_does_not_see_the_outer_pack(copier):
    outer_dst = _zeros(2)
    inner_dst = _zeros(2)
    other = _new_copier()

    device_copy.begin()
    _copy(outer_dst, _host([1, 2]))
    with device_copy.active(other):
        assert not device_copy.is_open()
        _copy(inner_dst, _host([3, 4]))  # plain copy: the inner pack is closed
        device_copy.begin()
        _copy(inner_dst, _host([5, 6]))
        torch.cuda.synchronize()
        assert inner_dst.tolist() == [3, 4]
        assert outer_dst.tolist() == [0, 0]
        device_copy.flush()
    assert device_copy.is_open() and len(copier._pack.h2d) == 1
    device_copy.flush()
    torch.cuda.synchronize()

    assert inner_dst.tolist() == [5, 6]
    assert outer_dst.tolist() == [1, 2]


def test_active_none_disables_the_copier_inside_an_outer_one(copier):
    dst = _zeros(2)
    values = _host([7, 8])
    assert _skip_copy(dst, values) is True

    device_copy.begin()
    with device_copy.active(None):
        assert device_copy.current() is None and not device_copy.is_open()
        assert _skip_copy(dst, values) is True
        torch.cuda.synchronize()
        assert dst.tolist() == [7, 8]
    assert copier._pack.h2d == []
    assert _skip_copy(dst, values) is False


def test_two_copiers_do_not_share_shadows(copier):
    dst = _zeros(3)
    values = _host([7, 8, 9])

    assert _skip_copy(dst, values) is True
    with device_copy.active(_new_copier()):
        # The other copier has never seen dst.
        assert _skip_copy(dst, values) is True
        assert _skip_copy(dst, values) is False
    assert _skip_copy(dst, values) is False


def test_a_copier_active_on_one_thread_is_invisible_to_another(copier):
    mine = _zeros(2)
    theirs = _zeros(2)
    seen = {}

    device_copy.begin()
    _copy(mine, _host([1, 1]))

    def other_thread():
        seen["open"] = device_copy.is_open()
        seen["copier"] = device_copy.current()
        _copy(theirs, _host([2, 2]))
        device_copy.flush()
        torch.cuda.synchronize()
        seen["theirs"] = theirs.tolist()
        seen["mine"] = mine.tolist()

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()

    assert seen == {"open": False, "copier": None, "theirs": [2, 2], "mine": [0, 0]}
    assert device_copy.is_open() and len(copier._pack.h2d) == 1
    device_copy.flush()
    torch.cuda.synchronize()
    assert mine.tolist() == [1, 1]


def test_threads_with_their_own_copiers_keep_them_apart():
    copiers = [_new_copier() for _ in range(2)]
    dsts = [_zeros(4) for _ in range(2)]
    barrier = threading.Barrier(2)
    results = {}

    def worker(index):
        with device_copy.active(copiers[index]):
            device_copy.begin()
            _copy(dsts[index], _host([index + 1] * 4))
            barrier.wait()
            # Both packs are open with one staged copy each.
            results[index] = (copiers[index].is_open, len(copiers[index]._pack.h2d))
            barrier.wait()
            device_copy.flush()
            torch.cuda.synchronize()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {0: (True, 1), 1: (True, 1)}
    assert dsts[0].tolist() == [1] * 4
    assert dsts[1].tolist() == [2] * 4
