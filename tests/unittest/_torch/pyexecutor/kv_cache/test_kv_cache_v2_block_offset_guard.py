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
"""KVCacheManagerV2's opt-in unchanged block-table skip (``skip_unchanged_block_offsets``).

A model arms it on a separate draft KV cache manager through RuntimeHooks
(``skip_unchanged_draft_block_offsets``).

The guard is driven on a manager built without ``__init__`` (every attribute it reads is declared on the class or set
here). The CUDA test checks the bytes: a guarded sequence of copies (the in-tree copy op issued only when the guard
does not skip) leaves the destination equal to always copying, after every step of a random host-table history.
"""

import random

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

POOLS, CAPACITY, WIDTH = 2, 6, 16


def _manager(pin: bool = False) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.host_kv_cache_block_offsets = torch.zeros(
        POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32, pin_memory=pin
    )
    manager.host_kv_cache_block_offsets[:, :, 0, :] = torch.arange(
        CAPACITY * WIDTH, dtype=torch.int32
    ).view(CAPACITY, WIDTH)
    manager.skip_unchanged_block_offsets = True
    return manager


def _rows(*rows: int) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32)


@pytest.mark.cpu_only
def test_the_guard_is_off_by_default():
    assert KVCacheManagerV2.skip_unchanged_block_offsets is False
    assert KVCacheManagerV2.patch_unchanged_block_offsets is False


@pytest.mark.cpu_only
def test_an_unchanged_decode_copy_is_skipped():
    manager, dst = _manager(), torch.zeros(POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32)

    assert not manager._block_offsets_unchanged(dst, _rows(3), 0)
    assert manager._block_offsets_unchanged(dst, _rows(3), 0)
    assert manager._block_offsets_unchanged(dst, _rows(3), 0)
    assert (manager.block_offset_copies, manager.block_offset_skips) == (1, 2)


@pytest.mark.cpu_only
@pytest.mark.parametrize("pool", range(POOLS))
def test_a_changed_page_index_in_a_copied_row_copies(pool):
    manager, dst = _manager(), torch.zeros(POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32)
    assert not manager._block_offsets_unchanged(dst, _rows(1, 3), 0)

    manager.host_kv_cache_block_offsets[pool, 3, 0, WIDTH - 1] += 1
    assert not manager._block_offsets_unchanged(dst, _rows(1, 3), 0)
    assert manager._block_offsets_unchanged(dst, _rows(1, 3), 0)


@pytest.mark.cpu_only
def test_field_one_and_rows_not_copied_do_not_force_a_copy():
    """The copy reads field 0 of the copied rows only."""
    manager, dst = _manager(), torch.zeros(POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32)
    assert not manager._block_offsets_unchanged(dst, _rows(2), 0)

    manager.host_kv_cache_block_offsets[:, 2, 1, :] = 7
    manager.host_kv_cache_block_offsets[:, 4, 0, :] = 9
    assert manager._block_offsets_unchanged(dst, _rows(2), 0)


@pytest.mark.cpu_only
def test_another_row_set_or_order_copies():
    manager, dst = _manager(), torch.zeros(POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32)
    assert not manager._block_offsets_unchanged(dst, _rows(1, 3), 0)

    assert not manager._block_offsets_unchanged(dst, _rows(3, 1), 0)
    assert not manager._block_offsets_unchanged(dst, _rows(3), 0)
    assert manager._block_offsets_unchanged(dst, _rows(3), 0)


@pytest.mark.cpu_only
def test_a_step_with_context_requests_copies_and_drops_the_record():
    manager, dst = _manager(), torch.zeros(POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32)
    assert not manager._block_offsets_unchanged(dst, _rows(3), 0)

    assert not manager._block_offsets_unchanged(dst, _rows(3), 1)
    assert not manager._block_offsets_unchanged(dst, _rows(3), 0)
    assert manager._block_offsets_unchanged(dst, _rows(3), 0)


@pytest.mark.cpu_only
def test_another_view_of_the_same_storage_copies_and_replaces_the_record():
    manager = _manager()
    storage = torch.zeros(POOLS, CAPACITY + 2, 2, WIDTH, dtype=torch.int32)
    whole, prefix = storage, storage[:, :CAPACITY]
    assert not manager._block_offsets_unchanged(whole, _rows(3), 0)

    assert not manager._block_offsets_unchanged(prefix, _rows(3), 0)
    assert not manager._block_offsets_unchanged(whole, _rows(3), 0)
    assert manager._block_offsets_unchanged(whole, _rows(3), 0)


@pytest.mark.cpu_only
def test_destinations_in_different_storages_keep_their_own_records():
    manager = _manager()
    first, second = (torch.zeros(POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32) for _ in range(2))
    assert not manager._block_offsets_unchanged(first, _rows(3), 0)
    assert not manager._block_offsets_unchanged(second, _rows(3), 0)

    assert manager._block_offsets_unchanged(first, _rows(3), 0)
    assert manager._block_offsets_unchanged(second, _rows(3), 0)


@pytest.mark.cpu_only
def test_the_counts_are_logged_at_powers_of_two(monkeypatch):
    lines = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.logger.debug", lines.append
    )
    manager, dst = _manager(), torch.zeros(POOLS, CAPACITY, 2, WIDTH, dtype=torch.int32)
    for _ in range(9):
        manager._block_offsets_unchanged(dst, _rows(3), 0)

    assert lines == [
        f"KVCacheManagerV2 block-offset copies skipped when unchanged: {skips} of {calls}"
        for skips, calls in ((0, 1), (1, 2), (3, 4), (7, 8))
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the in-tree block-offset copy op")
@pytest.mark.parametrize("seed", range(4))
def test_guarded_copies_leave_the_bytes_of_always_copying(seed):
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    rng = random.Random(seed)
    manager = _manager(pin=True)
    host = manager.host_kv_cache_block_offsets
    scales = torch.tensor([2, 3], dtype=torch.int32, pin_memory=True)
    offsets = torch.tensor([1, 5], dtype=torch.int32, pin_memory=True)
    guarded = torch.full((POOLS, CAPACITY, 2, WIDTH), -1, dtype=torch.int32, device="cuda")
    reference = guarded.clone()
    stream = torch.cuda.current_stream().cuda_stream
    rows = [1, 3]
    skips = 0
    for step in range(200):
        event = rng.random()
        if event < 0.15:
            host[rng.randrange(POOLS), rng.choice(rows), 0, rng.randrange(WIDTH)] = rng.randrange(
                1000
            )
        elif event < 0.25:
            host[rng.randrange(POOLS), rng.randrange(CAPACITY), 1, rng.randrange(WIDTH)] = (
                rng.randrange(1000)
            )
        elif event < 0.30:
            rows = rng.sample(range(CAPACITY), rng.randint(1, 3))
        num_contexts = 1 if rng.random() < 0.05 else 0
        copy_idx = torch.tensor(rows, dtype=torch.int32, pin_memory=True)
        copy_batch_block_offsets_to_device(host, reference, copy_idx, scales, offsets, stream)
        if manager._block_offsets_unchanged(guarded, copy_idx, num_contexts):
            skips += 1
        else:
            copy_batch_block_offsets_to_device(host, guarded, copy_idx, scales, offsets, stream)
        torch.cuda.synchronize()
        assert torch.equal(guarded, reference), f"seed {seed} step {step}"
    assert skips > 50
