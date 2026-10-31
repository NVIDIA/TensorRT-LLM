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


def test_failed_copy_through_another_view_invalidates_the_old_record(cache, monkeypatch):
    dst = _device_zeros(4)
    host = _host([1, 2, 3, 4])
    assert cache.copy(dst, host, pack=False)

    def partial_copy_then_fail(target, source):
        target.copy_(source, non_blocking=True)
        raise RuntimeError("copy submission failed")

    monkeypatch.setattr(h2d_cache.step_inputs, "copy_", partial_copy_then_fail)
    with pytest.raises(RuntimeError, match="copy submission failed"):
        cache.copy(dst[1:], _host([9, 9, 9]), pack=True)
    assert cache.copy(dst, host, pack=False)
    torch.cuda.synchronize()
    assert dst.tolist() == host.tolist()
