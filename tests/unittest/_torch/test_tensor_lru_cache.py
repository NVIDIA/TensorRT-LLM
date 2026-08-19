# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tensorrt_llm._torch.tensor_lru_cache import (
    CacheAllocationResult,
    CacheEntryState,
    TensorLRUCache,
)


def test_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        TensorLRUCache[str](0)


def test_put_get_pop_and_clear_update_byte_accounting() -> None:
    cache = TensorLRUCache[str](max_bytes=32)
    tensor = torch.arange(4, dtype=torch.float32)

    assert cache.max_bytes == 32
    assert cache.get("missing") is None
    assert cache.put("a", tensor)
    cached = cache.get("a")

    assert len(cache) == 1
    assert cache.current_bytes == 16
    assert cached is not None
    assert cached is not tensor
    torch.testing.assert_close(cached, tensor)

    assert cache.pop("missing") is None
    assert cache.pop("a") is cached
    assert len(cache) == 0
    assert cache.current_bytes == 0

    assert cache.put("a", tensor)
    cache.clear()
    assert len(cache) == 0
    assert cache.current_bytes == 0
    assert cache.get("a") is None


def test_clear_rejects_live_references_and_reservations() -> None:
    cache = TensorLRUCache[str](max_bytes=32)

    assert cache.allocate("key", 8) is CacheAllocationResult.NEW_RESERVATION
    with pytest.raises(RuntimeError, match="live references or reservations"):
        cache.clear()

    assert cache.put(
        "key",
        torch.ones(2, dtype=torch.float32),
        expected_state=CacheEntryState.RESERVED,
    )
    with pytest.raises(RuntimeError, match="live references or reservations"):
        cache.clear()

    assert cache.release("key") is None
    cache.clear()
    assert len(cache) == 0


def test_stats_track_hits_misses_insertions_and_replacements() -> None:
    cache = TensorLRUCache[str](max_bytes=32)
    tensor = torch.ones(2, dtype=torch.float32)
    replacement = torch.zeros(2, dtype=torch.float32)

    assert cache.get("missing") is None
    assert cache.put("key", tensor)
    cached = cache.get("key")
    assert cached is not None
    torch.testing.assert_close(cached, tensor)
    assert cache.put("key", replacement)

    stats = cache.stats()
    assert stats.max_bytes == 32
    assert stats.current_bytes == 8
    assert stats.item_count == 1
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.insertions == 1
    assert stats.replacements == 1
    assert stats.evictions == 0
    assert stats.rejected_insertions == 0
    assert stats.hit_rate == 0.5


def test_stats_track_evictions_and_rejected_insertions() -> None:
    cache = TensorLRUCache[str](max_bytes=16)

    assert cache.put("first", torch.ones(2, dtype=torch.float32))
    assert cache.put("second", torch.ones(2, dtype=torch.float32))
    assert cache.put("third", torch.ones(2, dtype=torch.float32))
    assert not cache.put("oversized", torch.ones(5, dtype=torch.float32))

    stats = cache.stats()
    assert stats.current_bytes == 16
    assert stats.item_count == 2
    assert stats.insertions == 3
    assert stats.evictions == 1
    assert stats.rejected_insertions == 1
    assert cache.get("first") is None


def test_stats_returns_immutable_snapshot() -> None:
    cache = TensorLRUCache[str](max_bytes=16)

    stats = cache.stats()
    assert cache.get("missing") is None

    assert stats.misses == 0
    assert cache.stats().misses == 1


def test_get_promotes_entry_before_lru_eviction() -> None:
    cache = TensorLRUCache[str](max_bytes=16)
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0, 4.0])
    third = torch.tensor([5.0, 6.0])

    assert cache.put("first", first)
    assert cache.put("second", second)
    cached_first = cache.get("first")
    assert cached_first is not None
    torch.testing.assert_close(cached_first, first)
    assert cache.put("third", third)

    assert cache.get("second") is None
    assert cache.get("first") is cached_first
    cached_third = cache.get("third")
    assert cached_third is not None
    torch.testing.assert_close(cached_third, third)
    assert len(cache) == 2
    assert cache.current_bytes == 16


def test_replace_updates_size_and_oversized_replace_leaves_old_value() -> None:
    cache = TensorLRUCache[str](max_bytes=20)
    small = torch.ones(2, dtype=torch.float32)
    larger = torch.ones(3, dtype=torch.float32)
    oversized = torch.ones(6, dtype=torch.float32)

    assert cache.put("key", small)
    assert cache.current_bytes == 8
    assert cache.put("key", larger)
    assert cache.current_bytes == 12
    cached = cache.get("key")
    assert cached is not None
    torch.testing.assert_close(cached, larger)

    assert not cache.put("key", oversized)
    assert cache.get("key") is cached
    assert cache.current_bytes == 12


def test_insert_detaches_and_owns_inserted_tensor_content() -> None:
    cache = TensorLRUCache[str](max_bytes=32)
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)

    assert cache.put("key", tensor)
    cached = cache.get("key")
    assert cached is not None
    assert cached is not tensor
    assert not cached.requires_grad

    tensor.detach()[0] = 99.0
    torch.testing.assert_close(cached, torch.tensor([1.0, 2.0]))


def test_parallel_operations_keep_cache_metadata_consistent() -> None:
    cache = TensorLRUCache[int](max_bytes=64)

    def write_and_read(index: int) -> None:
        value = torch.full((2,), index, dtype=torch.float32)
        assert cache.put(index, value)
        hit = cache.get(index)
        if hit is not None:
            assert hit.shape == value.shape

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_and_read, range(64)))

    assert len(cache) <= 8
    assert cache.current_bytes <= cache.max_bytes
    for index in range(64):
        hit = cache.get(index)
        if hit is not None:
            assert hit.numel() * hit.element_size() == 8


def test_shared_reservation_stores_one_output_and_keeps_reusable_entry() -> None:
    cache = TensorLRUCache[str](max_bytes=16)
    value = torch.ones(2, dtype=torch.float32)

    assert cache.allocate("key", 8) is CacheAllocationResult.NEW_RESERVATION
    assert cache.allocate("key", 8) is CacheAllocationResult.RESERVATION_HIT
    assert cache.current_bytes == 0
    assert cache.stats().reserved_bytes == 8
    assert cache.stats().pinned_bytes == 0

    assert cache.make_space_for("key") == []
    assert cache.put("key", value, expected_state=CacheEntryState.RESERVED)
    assert cache.current_bytes == 8
    assert cache.stats().reserved_bytes == 0
    assert cache.stats().pinned_bytes == 8
    cached = cache.get("key", record_stats=False)
    assert cached is not None
    torch.testing.assert_close(cached, value)

    assert cache.release("key") is None
    assert cache.release("key") is None
    assert cache.stats().pinned_bytes == 0
    assert cache.current_bytes == 8
    assert cache.allocate("key", 8) is CacheAllocationResult.READY_HIT
    assert cache.release("key") is None

    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.producer_misses == 1
    assert stats.inflight_deduplications == 1


def test_non_retained_entries_are_removed_on_their_final_release() -> None:
    cache = TensorLRUCache[str](max_bytes=16)

    assert (
        cache.allocate("reserved", 8, retain_after_release=False)
        is CacheAllocationResult.NEW_RESERVATION
    )
    assert cache.release("reserved") is None
    assert len(cache) == 0
    assert cache.stats().reserved_bytes == 0

    assert (
        cache.allocate("ready", 8, retain_after_release=False)
        is CacheAllocationResult.NEW_RESERVATION
    )
    assert cache.put(
        "ready",
        torch.ones(2, dtype=torch.float32),
        expected_state=CacheEntryState.RESERVED,
    )
    assert cache.release("ready") == "ready"
    assert len(cache) == 0
    assert cache.current_bytes == 0
    assert cache.stats().pinned_bytes == 0


def test_reservation_limit_and_output_space_are_checked_separately() -> None:
    cache = TensorLRUCache[str](max_bytes=16)
    assert cache.put("old-1", torch.ones(2, dtype=torch.float32))
    assert cache.put("old-2", torch.ones(2, dtype=torch.float32))

    assert cache.allocate("new-1", 8) is CacheAllocationResult.NEW_RESERVATION
    assert cache.allocate("new-2", 8) is CacheAllocationResult.NEW_RESERVATION
    assert cache.allocate("old-1", 8) is None
    assert cache.stats().blocked_allocations == 1

    assert cache.make_space_for("new-1") == ["old-1"]
    assert cache.make_space_for("new-2", pending_output_bytes=8) == ["old-2"]
    assert cache.current_bytes == 0
    assert cache.put(
        "new-1",
        torch.ones(2, dtype=torch.float32),
        expected_state=CacheEntryState.RESERVED,
    )
    assert cache.put(
        "new-2",
        torch.ones(2, dtype=torch.float32),
        expected_state=CacheEntryState.RESERVED,
    )
    assert cache.current_bytes == 16


def test_remote_cache_put_and_removal_require_the_expected_state() -> None:
    cache = TensorLRUCache[str](max_bytes=16)
    value = torch.ones(2, dtype=torch.float32)

    with pytest.raises(ValueError, match="requires expected_bytes"):
        cache.put("key", value, expected_state=CacheEntryState.ABSENT)
    assert cache.put(
        "key",
        value,
        expected_state=CacheEntryState.ABSENT,
        expected_bytes=8,
    )
    with pytest.raises(RuntimeError, match="expected CacheEntryState.ABSENT"):
        cache.put(
            "key",
            value,
            expected_state=CacheEntryState.ABSENT,
            expected_bytes=8,
        )

    popped = cache.pop("key", expected_state=CacheEntryState.READY)
    assert popped is not None
    torch.testing.assert_close(popped, value)
    with pytest.raises(RuntimeError, match="target is absent"):
        cache.pop("key", expected_state=CacheEntryState.READY)


def test_stream_aware_mode_leaves_cpu_cache_behavior_unchanged() -> None:
    cache = TensorLRUCache[str](max_bytes=16, cuda_stream_aware=True)
    source = torch.arange(4, dtype=torch.float32)

    assert cache.put("key", source)
    cached = cache.get("key")

    assert cached is not None
    assert cached.device.type == "cpu"
    torch.testing.assert_close(cached, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_stream_aware_cache_orders_cross_stream_producer_and_consumer() -> None:
    cache = TensorLRUCache[str](max_bytes=16, cuda_stream_aware=True)
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    source = torch.zeros(4, device="cuda")

    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(10_000_000)
        source.fill_(7)
        assert cache.put("key", source)

    with torch.cuda.stream(consumer_stream):
        cached = cache.get("key")
        assert cached is not None
        observed = cached.clone()

    consumer_stream.synchronize()
    torch.testing.assert_close(observed, torch.full_like(observed, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_stream_aware_cache_pop_orders_cross_stream_consumer() -> None:
    cache = TensorLRUCache[str](max_bytes=16, cuda_stream_aware=True)
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    source = torch.zeros(4, device="cuda")

    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(10_000_000)
        source.fill_(5)
        assert cache.put("key", source)

    with torch.cuda.stream(consumer_stream):
        popped = cache.pop("key")
        assert popped is not None
        observed = popped.clone()

    consumer_stream.synchronize()
    assert len(cache) == 0
    torch.testing.assert_close(observed, torch.full_like(observed, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_stream_aware_cache_preserves_evicted_consumer_storage() -> None:
    cache = TensorLRUCache[str](max_bytes=4 * 1024, cuda_stream_aware=True)
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    source = torch.full((1024,), 3.0, device="cuda")

    with torch.cuda.stream(producer_stream):
        assert cache.put("key", source)

    with torch.cuda.stream(consumer_stream):
        cached = cache.get("key")
        assert cached is not None
        torch.cuda._sleep(10_000_000)
        observed = cached.clone()
    del cached

    with torch.cuda.stream(producer_stream):
        assert cache.put("other", torch.full_like(source, 9.0))

    stats = cache.stats()
    assert stats.evictions == 1
    assert stats.replacements == 0

    consumer_stream.synchronize()
    torch.testing.assert_close(observed, torch.full_like(observed, 3.0))
