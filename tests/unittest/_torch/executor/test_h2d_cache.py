# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skipping host-to-device metadata copies whose values the destination already holds."""

import threading

import pytest
import torch

from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor import h2d_cache
from tensorrt_llm._torch.pyexecutor.h2d_cache import (
    H2DSkipCache,
    copy_h2d_if_changed,
    copy_h2d_pool_shared,
    fill_tail_if_changed,
    fill_tail_pool_shared,
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
# CUDA graph metadata pool: one block shared by every graph key
# ---------------------------------------------------------------------------

NUM_SLOTS = 8


def _graph_key_views(name):
    """Two graph keys' metadata views from one pool, as create_cuda_graph_metadata hands them out."""
    pool = Buffers()
    large = pool.get_buffer([NUM_SLOTS], torch.int32, name, reserve_buffer=True)
    small = pool.get_buffer([NUM_SLOTS], torch.int32, name, reserve_buffer=True)
    assert small.data_ptr() == large.data_ptr()
    return large, small


def _capture_live_row_write(buf, rows, values):
    """A graph whose replay writes buf[:rows] on the device, as on_update_kv_lens writes the live slots."""
    slots = torch.tensor(values, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        buf[:rows].copy_(slots)
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def _key_b_step(buf, live):
    """Key B's prepare: its live slots, then -1 over the padded rows its captured producers also run."""
    copy_h2d_pool_shared(buf[: len(live)], _host(live))
    fill_tail_pool_shared(buf, len(live), -1)


@pytest.mark.parametrize("skip_pool_shared", [False, True], ids=["fixed", "kill_switch"])
def test_two_graph_keys_over_one_block_keep_the_padded_rows_unused(
    cache, monkeypatch, skip_pool_shared
):
    monkeypatch.setattr(h2d_cache, "SKIP_POOL_SHARED_UPLOADS", skip_pool_shared)
    key_a, key_b = _graph_key_views("msa_out_cache_loc")
    replay_a = _capture_live_row_write(key_a, 6, [10, 11, 12, 13, 14, 15])

    _key_b_step(key_b, [20, 21, 22])
    torch.cuda.synchronize()
    assert key_b.tolist() == [20, 21, 22, -1, -1, -1, -1, -1]
    # Key A's step: its replay writes six live slots into the shared block.
    replay_a.replay()
    # Key B's next step, with the same live slots.
    _key_b_step(key_b, [20, 21, 22])
    torch.cuda.synchronize()
    if skip_pool_shared:
        # The old behaviour: both writes were skipped, so key B's live and padded rows hold key A's live slots.
        assert key_b.tolist() == [10, 11, 12, 13, 14, 15, -1, -1]
    else:
        assert key_b.tolist() == [20, 21, 22, -1, -1, -1, -1, -1]


@pytest.mark.parametrize("skip_pool_shared", [False, True], ids=["fixed", "kill_switch"])
def test_two_graph_keys_over_one_block_keep_their_uploads(cache, monkeypatch, skip_pool_shared):
    monkeypatch.setattr(h2d_cache, "SKIP_POOL_SHARED_UPLOADS", skip_pool_shared)
    key_a, key_b = _graph_key_views("msa_block_table")
    replay_a = _capture_live_row_write(key_a, 4, [7, 7, 7, 7])
    table = [1, 2, 3, 4]

    assert copy_h2d_pool_shared(key_b[:4], _host(table)) is True
    replay_a.replay()
    copied = copy_h2d_pool_shared(key_b[:4], _host(table))
    torch.cuda.synchronize()
    if skip_pool_shared:
        assert copied is False
        assert key_b[:4].tolist() == [7, 7, 7, 7]
    else:
        assert copied is True
        assert key_b[:4].tolist() == table


def test_pool_shared_writes_drop_the_shadow_and_tail_record(cache):
    buf = _device_zeros(6)
    values = _host([1, 2])
    assert copy_h2d_if_changed(buf[:2], values) is True
    fill_tail_if_changed(buf, 2, -1)
    copy_h2d_pool_shared(buf[:2], _host([5, 6]))
    buf[3] = 9
    # Neither record vouches for the buffer after a write outside the skip cache.
    assert copy_h2d_if_changed(buf[:2], values) is True
    fill_tail_if_changed(buf, 2, -1)
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 2, -1, -1, -1, -1]


def test_pool_shared_writes_without_an_active_cache_are_plain_writes():
    assert not h2d_cache.enabled()
    buf = _device_zeros(4)
    assert copy_h2d_pool_shared(buf[:2], _host([3, 4]), pack=True) is True
    fill_tail_pool_shared(buf, 2, -1)
    torch.cuda.synchronize()
    assert buf.tolist() == [3, 4, -1, -1]
