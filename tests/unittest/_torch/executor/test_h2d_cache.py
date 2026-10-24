# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skipping host-to-device metadata copies whose values the destination already holds."""

import threading

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import h2d_cache
from tensorrt_llm._torch.pyexecutor.h2d_cache import (
    H2DSkipCache,
    copy_h2d_if_changed,
    fill_tail_if_changed,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def cache():
    """A fresh cache, active for the test body."""
    skip_cache = H2DSkipCache()
    with h2d_cache.active(skip_cache):
        yield skip_cache
    torch.cuda.synchronize()


def _host(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype).pin_memory()


def _device_zeros(n, dtype=torch.int32):
    return torch.zeros(n, dtype=dtype, device="cuda")


def test_without_an_active_cache_every_copy_is_issued_like_copy_():
    assert not h2d_cache.enabled()
    dst = _device_zeros(4)
    reference = _device_zeros(4)
    src = _host([3, 1, 4, 1])

    for _ in range(3):
        assert copy_h2d_if_changed(dst, src) is True
        reference.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(dst, reference)


def test_active_scopes_the_cache_to_the_with_block():
    skip_cache = H2DSkipCache()
    dst = _device_zeros(2)
    values = _host([1, 2])

    with h2d_cache.active(skip_cache):
        assert h2d_cache.enabled()
        assert copy_h2d_if_changed(dst, values) is True
        assert copy_h2d_if_changed(dst, values) is False
    assert not h2d_cache.enabled()
    assert copy_h2d_if_changed(dst, values) is True
    # Re-activating the same cache finds its shadow again.
    with h2d_cache.active(skip_cache):
        assert copy_h2d_if_changed(dst, values) is False


def test_active_none_disables_skipping_inside_an_outer_cache(cache):
    dst = _device_zeros(2)
    values = _host([1, 2])
    assert copy_h2d_if_changed(dst, values) is True

    with h2d_cache.active(None):
        assert not h2d_cache.enabled()
        assert copy_h2d_if_changed(dst, values) is True
    assert copy_h2d_if_changed(dst, values) is False


def test_two_caches_do_not_share_shadows(cache):
    dst = _device_zeros(3)
    values = _host([7, 8, 9])
    other = H2DSkipCache()

    assert copy_h2d_if_changed(dst, values) is True
    with h2d_cache.active(other):
        # The other cache has never seen dst.
        assert copy_h2d_if_changed(dst, values) is True
        assert copy_h2d_if_changed(dst, values) is False
    assert copy_h2d_if_changed(dst, values) is False


def test_a_cache_active_on_one_thread_is_invisible_to_another(cache):
    dst = _device_zeros(3)
    values = _host([4, 5, 6])
    assert copy_h2d_if_changed(dst, values) is True
    seen = {}

    def other_thread():
        seen["enabled"] = h2d_cache.enabled()
        seen["copied"] = [copy_h2d_if_changed(dst, values) for _ in range(2)]
        torch.cuda.synchronize()

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()

    assert seen == {"enabled": False, "copied": [True, True]}
    assert copy_h2d_if_changed(dst, values) is False


def test_threads_with_their_own_caches_keep_them_apart():
    dst = [_device_zeros(2) for _ in range(2)]
    values = _host([1, 1])
    caches = [H2DSkipCache(), H2DSkipCache()]
    barrier = threading.Barrier(2)
    results = {}

    def worker(index):
        with h2d_cache.active(caches[index]):
            first = copy_h2d_if_changed(dst[index], values)
            barrier.wait()
            second = copy_h2d_if_changed(dst[index], values)
            # The other thread's destination was never shadowed by this cache.
            other = copy_h2d_if_changed(dst[1 - index], values)
            torch.cuda.synchronize()
        results[index] = (first, second, other)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {0: (True, False, True), 1: (True, False, True)}


def test_a_repeated_identical_copy_is_skipped(cache):
    dst = _device_zeros(4)

    assert copy_h2d_if_changed(dst, _host([5, 6, 7, 8])) is True
    assert copy_h2d_if_changed(dst, _host([5, 6, 7, 8])) is False
    torch.cuda.synchronize()
    assert dst.tolist() == [5, 6, 7, 8]


def test_clear_makes_the_next_identical_copy_issue(cache):
    """The runner clears the cache after a discarded step, whose copies never landed."""
    dst = _device_zeros(3)
    values = _host([1, 2, 3])
    assert copy_h2d_if_changed(dst, values) is True

    cache.clear()

    assert copy_h2d_if_changed(dst, values) is True
    assert copy_h2d_if_changed(dst, values) is False
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3]


def test_changed_values_are_copied(cache):
    dst = _device_zeros(4)

    assert copy_h2d_if_changed(dst, _host([1, 2, 3, 4])) is True
    assert copy_h2d_if_changed(dst, _host([1, 2, 3, 5])) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 5]
    assert copy_h2d_if_changed(dst, _host([1, 2, 3, 5])) is False


def test_the_shadow_is_a_snapshot_of_the_source(cache):
    """Mutating the host source after the copy must not make a later copy look unchanged."""
    dst = _device_zeros(3)
    src = _host([1, 2, 3])

    assert copy_h2d_if_changed(dst, src) is True
    torch.cuda.synchronize()
    src[0] = 9
    assert copy_h2d_if_changed(dst, src) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [9, 2, 3]


def test_a_source_dtype_change_copies(cache):
    dst = _device_zeros(3)

    assert copy_h2d_if_changed(dst, _host([1, 2, 3], torch.int32)) is True
    assert copy_h2d_if_changed(dst, _host([1, 2, 3], torch.int64)) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3]


def test_a_copy_through_another_view_replaces_the_shadow(cache):
    buf = _device_zeros(8)
    head, tail = buf[:4], buf[4:]
    values = _host([1, 2, 3, 4])

    assert copy_h2d_if_changed(head, values) is True
    assert copy_h2d_if_changed(tail, _host([5, 6, 7, 8])) is True
    # The shadow now describes the tail, so the head copy must not be skipped.
    assert copy_h2d_if_changed(head, values) is True
    assert copy_h2d_if_changed(head, values) is False
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_an_overlapping_view_never_skips_on_stale_state(cache):
    buf = _device_zeros(6)
    values = _host([1, 1, 1, 1])

    assert copy_h2d_if_changed(buf[:4], values) is True
    assert copy_h2d_if_changed(buf[2:6], _host([2, 2, 2, 2])) is True
    # buf[2:4] was overwritten through the other view: same values, same view, must copy.
    assert copy_h2d_if_changed(buf[:4], values) is True
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 1, 1, 1, 2, 2]


def test_a_view_with_another_shape_of_the_same_bytes_copies(cache):
    buf = _device_zeros(4)

    assert copy_h2d_if_changed(buf, _host([1, 2, 3, 4])) is True
    assert copy_h2d_if_changed(buf.view(2, 2), _host([[1, 2], [3, 4]])) is True
    assert copy_h2d_if_changed(buf.view(2, 2), _host([[1, 2], [3, 4]])) is False


def test_a_cuda_source_always_copies_and_drops_the_shadow(cache):
    dst = _device_zeros(4)
    values = _host([1, 2, 3, 4])
    device_src = torch.arange(4, dtype=torch.int32, device="cuda")

    assert copy_h2d_if_changed(dst, values) is True
    assert copy_h2d_if_changed(dst, device_src) is True
    assert copy_h2d_if_changed(dst, device_src) is True
    # The device-side write replaced what the shadow described.
    assert copy_h2d_if_changed(dst, values) is True
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4]


def test_a_host_destination_always_copies(cache):
    dst = torch.zeros(3, dtype=torch.int32)

    assert copy_h2d_if_changed(dst, _host([1, 2, 3])) is True
    assert copy_h2d_if_changed(dst, _host([1, 2, 3])) is True
    assert dst.tolist() == [1, 2, 3]


def test_capture_always_copies_and_drops_the_shadow(cache):
    dst = _device_zeros(4)
    values = _host([4, 3, 2, 1])
    assert copy_h2d_if_changed(dst, values) is True
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    results = []
    with torch.cuda.graph(graph):
        results.append(copy_h2d_if_changed(dst, values))
    assert results == [True]

    dst.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert dst.tolist() == [4, 3, 2, 1]
    # The captured copy reruns on every replay, so the shadow cannot vouch for the buffer.
    assert copy_h2d_if_changed(dst, values) is True


def test_without_an_active_cache_every_tail_fill_runs():
    buf = _device_zeros(6)
    fill_tail_if_changed(buf, 2, -1)
    buf[3] = 5
    fill_tail_if_changed(buf, 2, -1)
    fill_tail_if_changed(buf, 6, -1)
    assert buf.tolist() == [0, 0, -1, -1, -1, -1]


def test_a_repeated_tail_fill_is_skipped_and_a_moved_start_refills(cache):
    buf = _device_zeros(6)
    fill_tail_if_changed(buf, 2, -1)
    # Writes nothing else is allowed to make; they show which fills ran.
    buf[3] = 5
    fill_tail_if_changed(buf, 2, -1)
    assert buf[3].item() == 5
    fill_tail_if_changed(buf, 4, -1)
    buf[3] = 7
    fill_tail_if_changed(buf, 2, -1)
    assert buf.tolist() == [0, 0, -1, -1, -1, -1]


def test_a_whole_buffer_write_drops_the_tail_record(cache):
    buf = _device_zeros(6)
    fill_tail_if_changed(buf, 2, -1)
    buf.copy_(torch.arange(6, dtype=torch.int32, device="cuda"))
    fill_tail_if_changed(buf, 6, -1)
    fill_tail_if_changed(buf, 2, -1)
    assert buf.tolist() == [0, 1, -1, -1, -1, -1]


def test_clear_drops_the_tail_records(cache):
    buf = _device_zeros(6)
    fill_tail_if_changed(buf, 2, -1)
    buf[4] = 9
    cache.clear()
    fill_tail_if_changed(buf, 2, -1)
    assert buf[2:].eq(-1).all()


def test_a_tail_record_keeps_its_storage_so_a_new_buffer_cannot_inherit_it(cache):
    buf = _device_zeros(1 << 20)
    fill_tail_if_changed(buf, 3, -1)
    address = buf.data_ptr()
    del buf
    # Without the record holding the storage, the allocator hands this same block back, and the fill below would be
    # skipped on the freed buffer's record.
    fresh = _device_zeros(1 << 20)
    assert fresh.data_ptr() != address
    fill_tail_if_changed(fresh, 3, -1)
    assert fresh[3:].eq(-1).all()


# ---------------------------------------------------------------------------
# changed_span (RuntimeHooks.h2d_changed_span): a changed packed 2-D upload stages only each row's changed span
# ---------------------------------------------------------------------------

ROWS, WIDTH = 4, 1024


def _span_pack():
    from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack

    return StepInputPack(torch.device("cuda"), parallel_scatter=True)


def _table(seed, rows=ROWS, width=WIDTH):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 1 << 20, (rows, width), generator=generator, dtype=torch.int32
    ).pin_memory()


def _upload(pack, span_cache, dst, src):
    """One step: open the pack, route the copy through the cache, issue the pack; returns the staged copies."""
    from tensorrt_llm._torch.pyexecutor import step_inputs

    with h2d_cache.active(span_cache), step_inputs.active(pack):
        step_inputs.begin()
        copied = copy_h2d_if_changed(dst, src, pack=True)
        staged = list(pack.h2d)
        step_inputs.flush()
    torch.cuda.synchronize()
    return copied, staged


def test_changed_span_is_off_by_default_and_without_init():
    assert not H2DSkipCache().changed_span
    assert not object.__new__(H2DSkipCache).changed_span


@pytest.mark.parametrize("changed_span", [False, True])
def test_changed_span_device_rows_equal_the_source_after_every_step(changed_span):
    """Bitwise: whatever is staged, the device view holds the source after each step."""
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=changed_span)
    buf = torch.full((8, 2 * WIDTH), -7, dtype=torch.int32, device="cuda")
    dst = buf[:ROWS, :WIDTH]
    src = _table(0)
    generator = torch.Generator().manual_seed(1)
    for step in range(40):
        src = src.clone().pin_memory()
        # A request crosses into a new page: one entry of one row changes (sometimes several, in several rows).
        for _ in range(1 + step % 3):
            row = int(torch.randint(0, ROWS, (1,), generator=generator))
            col = int(torch.randint(0, WIDTH, (1,), generator=generator))
            src[row, col] = step
        _upload(pack, span_cache, dst, src)
        assert torch.equal(dst.cpu(), src)
    # Rows and columns outside the view were never written.
    assert bool((buf[ROWS:] == -7).all()) and bool((buf[:, WIDTH:] == -7).all())
    assert (span_cache.span_uploads > 0) is changed_span


def test_changed_span_one_changed_entry_stages_one_element():
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(2)
    _upload(pack, span_cache, dst, src)

    src = src.clone().pin_memory()
    src[2, 517] += 1
    copied, staged = _upload(pack, span_cache, dst, src)

    assert copied is True
    assert staged == [(dst[2, 517].data_ptr(), staged[0][1], 4)]
    assert torch.equal(dst.cpu(), src)
    assert (span_cache.span_uploads, span_cache.span_bytes) == (1, 4)
    assert span_cache.span_full_bytes == ROWS * WIDTH * 4


def test_changed_span_each_changed_row_stages_its_first_to_last_changed_column():
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(3)
    _upload(pack, span_cache, dst, src)

    src = src.clone().pin_memory()
    src[0, 10] += 1
    src[0, 20] += 1
    src[3, 1000] += 1
    _, staged = _upload(pack, span_cache, dst, src)

    assert [(ptr, nbytes) for ptr, _, nbytes in staged] == [
        (dst[0, 10].data_ptr(), 11 * 4),
        (dst[3, 1000].data_ptr(), 4),
    ]
    assert torch.equal(dst.cpu(), src)


def test_changed_span_empty_span_stages_nothing_and_keeps_the_device_values():
    """PR-CTX (d): an empty span. A narrower view of the same rows whose values all equal the shadow's: no column
    changed, so nothing is staged (the copy still counts as issued, and the shadow follows the new view)."""
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    buf = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(10)
    _upload(pack, span_cache, buf, src)

    narrow = src[:, : WIDTH // 2].contiguous().pin_memory()
    copied, staged = _upload(pack, span_cache, buf[:, : WIDTH // 2], narrow)

    assert copied is True
    assert staged == []
    assert torch.equal(buf.cpu(), src)
    assert (span_cache.span_uploads, span_cache.span_bytes) == (1, 0)
    # The shadow is the narrow view now: the same values again are skipped.
    copied, staged = _upload(pack, span_cache, buf[:, : WIDTH // 2], narrow)
    assert copied is False and staged == []


def test_changed_span_whole_buffer_span_is_staged_whole():
    """PR-CTX (d): a whole-buffer span. The first and last column of every row changed, so every row's span is the
    whole row: over the fraction, the view is staged whole (one descriptor), not as per-row spans."""
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(11)
    _upload(pack, span_cache, dst, src)

    src = src.clone().pin_memory()
    src[:, 0] += 1
    src[:, -1] += 1
    _, staged = _upload(pack, span_cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]
    assert torch.equal(dst.cpu(), src)
    assert span_cache.span_uploads == 0


def test_changed_span_at_the_fraction_is_still_a_span():
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(12)
    _upload(pack, span_cache, dst, src)

    src = src.clone().pin_memory()
    half = int(H2DSkipCache.CHANGED_SPAN_MAX_FRACTION * WIDTH)
    src[:, :half] += 1  # exactly the fraction of the view's elements
    _, staged = _upload(pack, span_cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [half * 4] * ROWS
    assert torch.equal(dst.cpu(), src)


def test_changed_span_wider_view_uploads_the_columns_the_shadow_does_not_cover():
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    buf = torch.full((ROWS, WIDTH), -1, dtype=torch.int32, device="cuda")
    src = _table(4)
    _upload(pack, span_cache, buf[:, : WIDTH // 2], src[:, : WIDTH // 2].contiguous().pin_memory())

    _, staged = _upload(
        pack,
        span_cache,
        buf[:, : WIDTH // 2 + 3],
        src[:, : WIDTH // 2 + 3].contiguous().pin_memory(),
    )

    assert [nbytes for _, _, nbytes in staged] == [3 * 4] * ROWS
    assert torch.equal(buf[:, : WIDTH // 2 + 3].cpu(), src[:, : WIDTH // 2 + 3])


def test_changed_span_a_mostly_changed_table_is_staged_whole():
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    _upload(pack, span_cache, dst, _table(5))

    src = _table(6)
    _, staged = _upload(pack, span_cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]
    assert torch.equal(dst.cpu(), src)


def test_changed_span_off_by_default_the_whole_view_is_staged():
    pack = _span_pack()
    span_cache = H2DSkipCache()
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(7)
    _upload(pack, span_cache, dst, src)

    src = src.clone().pin_memory()
    src[1, 1] += 1
    _, staged = _upload(pack, span_cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]


def test_changed_span_an_unpacked_copy_copies_the_whole_view():
    span_cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(8)
    with h2d_cache.active(span_cache):
        copy_h2d_if_changed(dst, src)
        src = src.clone().pin_memory()
        src[1, 1] += 1
        assert copy_h2d_if_changed(dst, src) is True
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), src)
    assert span_cache.span_uploads == 0


def test_changed_span_a_view_with_another_row_stride_is_staged_whole():
    """The shadow describes other device bytes: nothing of it may be trusted."""
    pack = _span_pack()
    span_cache = H2DSkipCache(changed_span=True)
    buf = torch.zeros((2 * ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(9)
    _upload(pack, span_cache, buf[:ROWS], src)

    _, staged = _upload(pack, span_cache, buf[::2], src)

    assert [nbytes for _, _, nbytes in staged] == [WIDTH * 4] * ROWS
    assert torch.equal(buf[::2].cpu(), src)


@pytest.mark.parametrize("changed_span", [False, True])
def test_changed_span_keeps_a_page_crossing_step_in_the_graph_window(changed_span):
    """A 16 KB page table (over the in-graph window) gains one page: with changed_span the step's pack still fits the
    window and the replay's prologue issues it; without, the step falls back to an eager upload and scatter."""
    from tensorrt_llm._torch.pyexecutor import step_inputs
    from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack

    pack = StepInputPack(torch.device("cuda"), parallel_scatter=True, in_graph=True)
    span_cache = H2DSkipCache(changed_span=changed_span)
    table = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    out = torch.zeros(ROWS, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with step_inputs.active(pack):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            assert step_inputs.capture_prologue() is pack
            out.copy_(table[:, -1])
    src = _table(13)
    table.copy_(src)
    with h2d_cache.active(span_cache):
        copy_h2d_if_changed(table, src)  # the shadow of the table the device holds
    torch.cuda.synchronize()
    src = src.clone().pin_memory()
    src[1, -1] = 4242
    with h2d_cache.active(span_cache), step_inputs.active(pack):
        step_inputs.begin()
        copy_h2d_if_changed(table, src, pack=True)
        step_inputs.flush(in_graph=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(table.cpu(), src)
    assert out.tolist() == src[:, -1].tolist()
    expect = (1, 0) if changed_span else (0, 1)
    assert (pack.graph_windows, pack.eager_fallbacks) == expect
