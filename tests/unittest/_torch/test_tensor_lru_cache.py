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

from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache


def test_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        TensorLRUCache[str](0)


def test_put_get_pop_and_clear_update_byte_accounting() -> None:
    cache = TensorLRUCache[str](max_bytes=32)
    tensor = torch.arange(4, dtype=torch.float32)

    assert cache.max_bytes == 32
    assert cache.get("missing") is None
    assert cache.put("a", tensor)

    assert len(cache) == 1
    assert cache.current_bytes == 16
    assert cache.get("a") is tensor

    assert cache.pop("missing") is None
    assert cache.pop("a") is tensor
    assert len(cache) == 0
    assert cache.current_bytes == 0

    assert cache.put("a", tensor)
    cache.clear()
    assert len(cache) == 0
    assert cache.current_bytes == 0
    assert cache.get("a") is None


def test_stats_track_hits_misses_insertions_and_replacements() -> None:
    cache = TensorLRUCache[str](max_bytes=32)
    tensor = torch.ones(2, dtype=torch.float32)
    replacement = torch.zeros(2, dtype=torch.float32)

    assert cache.get("missing") is None
    assert cache.put("key", tensor)
    assert cache.get("key") is tensor
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
    assert cache.get("first") is first
    assert cache.put("third", third)

    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third
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
    assert cache.get("key") is larger

    assert not cache.put("key", oversized)
    assert cache.get("key") is larger
    assert cache.current_bytes == 12


def test_clone_on_insert_detaches_and_owns_inserted_tensor_content() -> None:
    cache = TensorLRUCache[str](max_bytes=32, clone_on_insert=True)
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
