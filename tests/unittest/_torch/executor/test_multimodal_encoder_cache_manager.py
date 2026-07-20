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
import pytest
import torch

from tensorrt_llm._torch.multimodal_encoder_cache_manager import MultimodalEncoderCacheManager


def _tensor(rows: int) -> torch.Tensor:
    # 1 row == 8 bytes (2 fp32 elements) for easy byte math.
    return torch.full((rows, 2), 1.0, dtype=torch.float32)


ROW_BYTES = 8


def test_adopt_takes_ownership_without_clone():
    manager = MultimodalEncoderCacheManager(10 * ROW_BYTES)
    value = _tensor(4)
    stored = manager.adopt("k", value, request_id=1)
    assert stored.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    assert manager.current_bytes == 4 * ROW_BYTES
    assert manager.held_bytes == 4 * ROW_BYTES


def test_duplicate_adopt_collapses_to_one_entry():
    manager = MultimodalEncoderCacheManager(10 * ROW_BYTES)
    first = manager.adopt("k", _tensor(4), request_id=1)
    second = manager.adopt("k", _tensor(4), request_id=2)
    assert second is first  # single resident copy, both requests holding
    assert manager.current_bytes == 4 * ROW_BYTES
    assert len(manager) == 1
    assert manager.stats().dedup_adoptions == 1

    # Entry stays held until the *last* referencing request releases it.
    manager.release_holds(1)
    assert manager.held_bytes == 4 * ROW_BYTES
    manager.release_holds(2)
    assert manager.held_bytes == 0
    assert manager.current_bytes == 4 * ROW_BYTES  # still resident for reuse


def test_held_entries_are_never_evicted():
    manager = MultimodalEncoderCacheManager(10 * ROW_BYTES)
    manager.adopt("a", _tensor(6), request_id=1)
    manager.adopt("b", _tensor(4), request_id=1)

    assert not manager.can_allocate(1 * ROW_BYTES)
    with pytest.raises(RuntimeError, match="can_allocate"):
        manager.adopt("c", _tensor(1), request_id=2)
    assert manager.contains("a") and manager.contains("b")


def test_zero_ref_lru_eviction_makes_room():
    manager = MultimodalEncoderCacheManager(10 * ROW_BYTES)
    manager.adopt("old", _tensor(6), request_id=1)
    manager.adopt("new", _tensor(4), request_id=2)
    manager.release_holds(1)

    assert manager.can_allocate(6 * ROW_BYTES)
    manager.adopt("c", _tensor(6), request_id=3)
    assert not manager.contains("old")  # zero-ref LRU victim
    assert manager.contains("new") and manager.contains("c")
    assert manager.stats().evictions == 1


def test_get_and_hold_revives_freeable_entry():
    manager = MultimodalEncoderCacheManager(10 * ROW_BYTES)
    manager.adopt("k", _tensor(6), request_id=1)
    manager.release_holds(1)
    assert manager.held_bytes == 0

    hit = manager.get_and_hold("k", request_id=2)
    assert hit is not None
    assert manager.held_bytes == 6 * ROW_BYTES
    # Re-held entry is protected again.
    with pytest.raises(RuntimeError):
        manager.adopt("big", _tensor(5), request_id=3)


def test_get_and_hold_miss_returns_none():
    manager = MultimodalEncoderCacheManager(10 * ROW_BYTES)
    assert manager.get_and_hold("absent", request_id=1) is None
    assert manager.stats().misses == 1


def test_release_holds_is_idempotent_and_scoped():
    manager = MultimodalEncoderCacheManager(20 * ROW_BYTES)
    manager.adopt("a", _tensor(4), request_id=1)
    manager.adopt("b", _tensor(4), request_id=1)
    manager.adopt("c", _tensor(4), request_id=2)

    manager.release_holds(1)
    manager.release_holds(1)  # idempotent: cancel path may race normal strip
    manager.release_holds(99)  # unknown request is a no-op
    assert manager.held_bytes == 4 * ROW_BYTES  # only request 2's entry


def test_reserved_bytes_shrink_allocatable_space():
    manager = MultimodalEncoderCacheManager(10 * ROW_BYTES)
    assert manager.can_allocate(4 * ROW_BYTES, reserved_bytes=6 * ROW_BYTES)
    assert not manager.can_allocate(5 * ROW_BYTES, reserved_bytes=6 * ROW_BYTES)


def test_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        MultimodalEncoderCacheManager(0)
