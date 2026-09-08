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

"""Unit tests for locality domain localization support in _KVCache and KVCacheManager.

These tests isolate the localization behavior of _KVCache (memory node
affinity via locality_domain_id) and KVCacheManager (factory and query APIs) without
FakeEngine or full prefill/decode cycles.

Localized tests opt in with GpuCacheTierConfig(enable_locality_domains=True) and set
TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1 so the storage layer creates
GpuCacheLevelStorage with num_locality_domains=2. Tests that need the legacy
single-locality domain path keep enable_locality_domains=False so they stay non-localized even
on Rubin/B200 hardware.
"""

import gc
import os
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        _KVCache,
    )
    from kv_cache_manager_v2._common import (
        BAD_PAGE_INDEX,
        DEFAULT_BEAM_INDEX,
        GPU_LEVEL,
        CudaStream,
    )
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleId
    from kv_cache_manager_v2._storage._core import PoolGroupIndex, SlotId
    from kv_cache_manager_v2._utils import TemporaryCudaStream, init_cuda_once
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        _KVCache,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
        BAD_PAGE_INDEX,
        DEFAULT_BEAM_INDEX,
        GPU_LEVEL,
        CudaStream,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import LifeCycleId
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import PoolGroupIndex, SlotId
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import TemporaryCudaStream, init_cuda_once


# Canonical role names
ROLE_KEY = DataRole("key")
ROLE_VALUE = DataRole("value")

# 256 MiB — small enough for CI, large enough for multiple requests
_256MB = 256 * (1 << 20)
_TOKENS_PER_BLOCK = 32
_KV_BUF_SIZE = 8192  # 8 KB per KV buffer


def setUpModule() -> None:
    # Locality domains only exist in the Python backend; the C++ one reports a
    # single domain, so every assertion here would be meaningless.
    if os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower() != "python":
        raise unittest.SkipTest("locality domains require the Python KVCacheManagerV2 backend")


@contextmanager
def _override_localization_mode(mode: str):
    env_backup = os.environ.get("TRT_LLM_MOCK_LOCALIZATION_SUPPORT")
    try:
        os.environ["TRT_LLM_MOCK_LOCALIZATION_SUPPORT"] = mode
        yield
    finally:
        if env_backup is None:
            os.environ.pop("TRT_LLM_MOCK_LOCALIZATION_SUPPORT", None)
        else:
            os.environ["TRT_LLM_MOCK_LOCALIZATION_SUPPORT"] = env_backup


def _make_config(
    num_layers: int = 2,
    gpu_quota: int = _256MB,
    tokens_per_block: int = _TOKENS_PER_BLOCK,
    kv_buf_size: int = _KV_BUF_SIZE,
    enable_locality_domains: bool = False,
) -> KVCacheManagerConfig:
    """Build a simple KVCacheManagerConfig (GPU-only, no SWA)."""
    return KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        cache_tiers=[
            GpuCacheTierConfig(quota=gpu_quota, enable_locality_domains=enable_locality_domains)
        ],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(i),
                buffers=[
                    BufferConfig(role=ROLE_KEY, size=kv_buf_size),
                    BufferConfig(role=ROLE_VALUE, size=kv_buf_size),
                ],
                sliding_window_size=None,
            )
            for i in range(num_layers)
        ],
    )


def _make_localized_manager(**kwargs) -> KVCacheManager:
    """Create a KVCacheManager with mock localization enabled (num_locality_domains=2)."""
    kwargs.setdefault("enable_locality_domains", True)
    config = _make_config(**kwargs)
    with _override_localization_mode("1"):
        manager = KVCacheManager(config)
    return manager


def _make_non_localized_manager(**kwargs) -> KVCacheManager:
    """Create a KVCacheManager without localization (standard path)."""
    config = _make_config(**kwargs)
    with _override_localization_mode("0"):
        return KVCacheManager(config)


def _get_valid_page_indices(kv_cache: _KVCache, layer_group_id: int = 0) -> list[int]:
    """Extract non-BAD base page indices from a kv_cache."""
    indices = kv_cache.get_base_page_indices(layer_group_id)
    return [idx for idx in indices if idx != BAD_PAGE_INDEX]


def _get_valid_slot_ids(kv_cache: _KVCache, life_cycle_id: int = 0) -> list[int]:
    """Extract slot_ids from the active pages backing this KV cache."""
    slot_ids = []
    for block in kv_cache._blocks:
        block_page = block.pages[DEFAULT_BEAM_INDEX][life_cycle_id]
        if block_page is None:
            continue
        slot_ids.append(int(block_page.page.slot_id))
    return slot_ids


def _get_localized_slot_id_offset(kv_cache: _KVCache, layer_group_id: int = 0) -> int:
    storage = kv_cache.manager._storage._levels[GPU_LEVEL].storage
    return storage._pool_groups[PoolGroupIndex(layer_group_id)]._slot_id_offset


def _get_localized_base_page_index_offset(kv_cache: _KVCache, life_cycle_id: int = 0) -> int:
    storage = kv_cache.manager._storage
    slot_id_offset = _get_localized_slot_id_offset(kv_cache, life_cycle_id)
    return int(
        storage._base_page_index_for_slot(LifeCycleId(life_cycle_id), SlotId(slot_id_offset))
    )


class TestKVCacheLocalityDomainIdStorage(unittest.TestCase):
    """Tests that _KVCache correctly stores and exposes locality_domain_id."""

    manager: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = None

    def tearDown(self) -> None:
        gc.enable()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def test_locality_domain_id_stored_on_creation(self) -> None:
        """create_kv_cache(locality_domain_id=N) must store N in _KVCache.locality_domain_id."""
        self.manager = _make_localized_manager()
        kv0 = self.manager.create_kv_cache(locality_domain_id=0)
        kv1 = self.manager.create_kv_cache(locality_domain_id=1)
        self.assertEqual(kv0.locality_domain_id, 0)
        self.assertEqual(kv1.locality_domain_id, 1)
        kv0.close()
        kv1.close()

    def test_locality_domain_id_none_by_default(self) -> None:
        """create_kv_cache() without locality_domain_id must have locality_domain_id == None."""
        self.manager = _make_non_localized_manager()
        kv = self.manager.create_kv_cache()
        self.assertIsNone(kv.locality_domain_id)
        kv.close()

    def test_locality_domain_id_none_when_explicitly_passed(self) -> None:
        """create_kv_cache(locality_domain_id=None) on localized manager stores None."""
        self.manager = _make_localized_manager()
        kv = self.manager.create_kv_cache(locality_domain_id=None)
        self.assertIsNone(kv.locality_domain_id)
        kv.close()


class TestKVCacheLocalizedResize(unittest.TestCase):
    """Tests that _KVCache.resize() allocates slots from the correct locality domain."""

    manager: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = None

    def tearDown(self) -> None:
        gc.enable()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def _create_and_resize(self, locality_domain_id: int, capacity: int = 128) -> _KVCache:
        """Helper: create kv_cache, resume, resize to capacity."""
        kv_cache = self.manager.create_kv_cache(locality_domain_id=locality_domain_id)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = kv_cache.resume(stream)
            self.assertTrue(success, "resume() must succeed")
            success = kv_cache.resize(capacity)
            self.assertTrue(success, f"resize({capacity}) must succeed")
        return kv_cache

    def test_resize_slots_on_locality_domain0(self) -> None:
        """Slots allocated for locality_domain_id=0 must have page indices below the localized offset."""
        self.manager = _make_localized_manager()
        kv_cache = self._create_and_resize(locality_domain_id=0)
        indices = _get_valid_page_indices(kv_cache)
        localized_offset = _get_localized_base_page_index_offset(kv_cache)
        self.assertGreater(len(indices), 0, "Must have at least one allocated page")
        for idx in indices:
            self.assertLess(idx, localized_offset)
        kv_cache.close()

    def test_resize_slots_on_locality_domain1(self) -> None:
        """Slots allocated for locality_domain_id=1 must have page indices at or above the localized offset."""
        self.manager = _make_localized_manager()
        kv_cache = self._create_and_resize(locality_domain_id=1)
        indices = _get_valid_page_indices(kv_cache)
        localized_offset = _get_localized_base_page_index_offset(kv_cache)
        self.assertGreater(len(indices), 0, "Must have at least one allocated page")
        for idx in indices:
            self.assertGreaterEqual(idx, localized_offset)
        kv_cache.close()

    def test_resize_does_not_cross_locality_domain(self) -> None:
        """Two caches on different locality domains must have non-overlapping page index ranges."""
        self.manager = _make_localized_manager()
        kv0 = self._create_and_resize(locality_domain_id=0, capacity=64)
        kv1 = self._create_and_resize(locality_domain_id=1, capacity=64)

        indices0 = set(_get_valid_page_indices(kv0))
        indices1 = set(_get_valid_page_indices(kv1))
        slot_ids0 = set(_get_valid_slot_ids(kv0))
        slot_ids1 = set(_get_valid_slot_ids(kv1))
        localized_offset = _get_localized_base_page_index_offset(kv0)

        self.assertGreater(len(indices0), 0)
        self.assertGreater(len(indices1), 0)
        self.assertTrue(all(i < localized_offset for i in indices0))
        self.assertTrue(all(i >= localized_offset for i in indices1))
        self.assertEqual(
            indices0 & indices1, set(), "page index ranges must not overlap across locality domains"
        )
        self.assertEqual(
            slot_ids0 & slot_ids1, set(), "slot_id ranges must not overlap across locality domains"
        )

        kv0.close()
        kv1.close()


class TestKVCacheSuspendResumeLocalized(unittest.TestCase):
    """Tests that suspend/resume preserves locality domain affinity."""

    manager: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = None

    def tearDown(self) -> None:
        gc.enable()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def test_suspend_resume_preserves_locality_domain(self) -> None:
        """After suspend + resume, page indices must still stay on the original locality domain."""
        self.manager = _make_localized_manager()
        kv_cache = self.manager.create_kv_cache(locality_domain_id=1)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = kv_cache.resume(stream)
            self.assertTrue(success)
            success = kv_cache.resize(64)
            self.assertTrue(success)

            # Record indices before suspend
            indices_before = _get_valid_page_indices(kv_cache)
            self.assertGreater(len(indices_before), 0)

            # Suspend and resume
            kv_cache.suspend()
            self.assertEqual(kv_cache.status, _KVCache.Status.SUSPENDED)

        with TemporaryCudaStream([]) as s2:
            stream2 = cast(CudaStream, s2.handle)
            success = kv_cache.resume(stream2)
            self.assertTrue(success, "resume() after suspend must succeed")

            localized_offset = _get_localized_base_page_index_offset(kv_cache)
            # Page indices must still be on locality_domain1.
            indices_after = _get_valid_page_indices(kv_cache)
            self.assertGreater(len(indices_after), 0)
            for idx in indices_after:
                self.assertGreaterEqual(idx, localized_offset)

        kv_cache.close()


class TestKVCacheNonLocalizedBackwardCompat(unittest.TestCase):
    """Tests that locality_domain_id=None on a non-localized manager works identically to before."""

    manager: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = None

    def tearDown(self) -> None:
        gc.enable()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def test_non_localized_full_lifecycle(self) -> None:
        """Full create → resume → resize → commit → close on non-localized manager."""
        self.manager = _make_non_localized_manager()
        tokens = list(range(64))
        kv_cache = self.manager.create_kv_cache(input_tokens=tokens[:-1])
        self.assertIsNone(kv_cache.locality_domain_id)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = kv_cache.resume(stream)
            self.assertTrue(success)

            capacity = 128
            success = kv_cache.resize(capacity)
            self.assertTrue(success)
            self.assertEqual(kv_cache.capacity, capacity)

            # Commit some tokens
            uncommitted_start = kv_cache.num_committed_tokens
            new_tokens = tokens[uncommitted_start:]
            if new_tokens:
                kv_cache.commit(new_tokens)

            kv_cache.stop_committing()

            # Verify page indices exist and are valid.
            indices = _get_valid_page_indices(kv_cache)
            self.assertGreater(len(indices), 0)
            for idx in indices:
                self.assertGreaterEqual(idx, 0)

        kv_cache.close()
        self.assertEqual(kv_cache.status, _KVCache.Status.CLOSED)


class TestKVCacheManagerLocalizedQueries(unittest.TestCase):
    """Tests for KVCacheManager.num_locality_domains and get_per_locality_domain_free_slots()."""

    manager: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = None

    def tearDown(self) -> None:
        gc.enable()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def test_num_locality_domains_non_localized(self) -> None:
        """Non-localized manager must report num_locality_domains == 1."""
        self.manager = _make_non_localized_manager()
        self.assertEqual(self.manager.num_locality_domains, 1)

    def test_num_locality_domains_localized(self) -> None:
        """Localized manager must report num_locality_domains == 2."""
        self.manager = _make_localized_manager()
        self.assertEqual(self.manager.num_locality_domains, 2)

    def test_get_per_locality_domain_free_slots_initial(self) -> None:
        """Both locality domains should have equal free slot counts initially."""
        self.manager = _make_localized_manager()
        free_slots = self.manager.get_per_locality_domain_free_slots()
        self.assertEqual(len(free_slots), 2)
        self.assertGreater(free_slots[0], 0)
        self.assertEqual(free_slots[0], free_slots[1])

    def test_get_per_locality_domain_free_slots_after_allocation(self) -> None:
        """Allocating on locality domain-0 must decrease only locality domain-0's free slot count."""
        self.manager = _make_localized_manager()
        free_before = self.manager.get_per_locality_domain_free_slots()

        kv_cache = self.manager.create_kv_cache(locality_domain_id=0)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv_cache.resume(stream)
            kv_cache.resize(64)

        free_after = self.manager.get_per_locality_domain_free_slots()
        self.assertLess(
            free_after[0],
            free_before[0],
            "locality domain-0 free slots must decrease after allocation",
        )
        self.assertEqual(
            free_after[1], free_before[1], "locality domain-1 free slots must remain unchanged"
        )
        kv_cache.close()

    def test_get_per_locality_domain_free_slots_after_release(self) -> None:
        """Closing a kv_cache should restore free slots on its locality domain."""
        self.manager = _make_localized_manager()
        free_before = self.manager.get_per_locality_domain_free_slots()

        kv_cache = self.manager.create_kv_cache(locality_domain_id=1)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv_cache.resume(stream)
            kv_cache.resize(64)

        free_during = self.manager.get_per_locality_domain_free_slots()
        self.assertLess(free_during[1], free_before[1])

        kv_cache.close()
        free_after = self.manager.get_per_locality_domain_free_slots()
        self.assertEqual(free_after[1], free_before[1], "Free slots must be restored after close")

    def test_get_per_locality_domain_free_slots_non_localized(self) -> None:
        """Non-localized manager returns a single-element list."""
        self.manager = _make_non_localized_manager()
        free_slots = self.manager.get_per_locality_domain_free_slots()
        self.assertEqual(len(free_slots), 1)
        self.assertGreater(free_slots[0], 0)

    def test_page_index_upper_bound_covers_localized_indices(self) -> None:
        """Shared-pool-pointer buffers must span the displaced locality_domain1 page range."""
        self.manager = _make_localized_manager()
        kv_cache = self.manager.create_kv_cache(locality_domain_id=1)

        try:
            with TemporaryCudaStream([]) as s:
                stream = cast(CudaStream, s.handle)
                success = kv_cache.resume(stream)
                self.assertTrue(success)
                success = kv_cache.resize(64)
                self.assertTrue(success)

            storage = self.manager._storage
            lc_id = storage._layer_to_life_cycle_ids[LayerId(0)]
            page_indices = [
                int(storage.get_page_indices_for_slot(lc_id, SlotId(slot_id)))
                for slot_id in _get_valid_slot_ids(kv_cache, life_cycle_id=int(lc_id))
            ]
            self.assertGreater(len(page_indices), 0)

            upper_bound = self.manager.get_page_index_upper_bound(LayerId(0), DataRole("key"))
            self.assertGreater(
                upper_bound,
                max(page_indices),
                "get_page_index_upper_bound must cover localized locality_domain1 page indices",
            )
        finally:
            kv_cache.close()

    def test_page_index_upper_bound_ignores_page_stride_divisibility(self) -> None:
        """Localized upper bounds must use canonical slot offsets, not LOCALIZATION_OFFSET/page_stride."""
        from unittest.mock import patch

        self.manager = _make_localized_manager()

        with patch.object(type(self.manager), "get_page_stride", lambda _self, _layer_id, _role: 3):
            upper_bound = self.manager.get_page_index_upper_bound(LayerId(0), DataRole("key"))
            self.assertGreater(upper_bound, 0)


class TestKVCacheSaltedReuse(unittest.TestCase):
    """Tests that salted tree task IDs isolate reuse across locality domains.

    With locality_domain_id salting, requests on different locality domains produce different
    radix tree keys for the same tokens. This means:
    - Same tokens on same locality domain → reuse (num_committed_tokens > 0)
    - Same tokens on different locality domain → no reuse (num_committed_tokens == 0)
    - locality_domain_id=None (non-localized) → reuse works as before
    """

    manager: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = None

    def tearDown(self) -> None:
        gc.enable()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def _commit_and_close(self, kv_cache: _KVCache, tokens: list[int], capacity: int) -> None:
        """Helper: resume, resize, commit all tokens, stop committing, close."""
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = kv_cache.resume(stream)
            assert success
            success = kv_cache.resize(capacity)
            assert success
            uncommitted = tokens[kv_cache.num_committed_tokens :]
            if uncommitted:
                kv_cache.commit(uncommitted)
            kv_cache.stop_committing()
        kv_cache.close()

    def test_same_locality_domain_reuses_blocks(self) -> None:
        """Two requests on the same locality domain with identical tokens must reuse blocks."""
        self.manager = _make_localized_manager()
        tokens = list(range(64))
        capacity = 128

        # Request A on locality domain-0: commit tokens
        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=0)
        self._commit_and_close(kv_a, tokens, capacity)

        # Request B on locality domain-0: should find reusable blocks
        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=0)
        self.assertGreater(
            kv_b.num_committed_tokens, 0, "Same locality domain must reuse committed blocks"
        )
        self._commit_and_close(kv_b, tokens, capacity)

    def test_different_locality_domain_does_not_reuse_blocks(self) -> None:
        """Two requests on different locality domains with identical tokens must NOT reuse blocks."""
        self.manager = _make_localized_manager()
        tokens = list(range(64))
        capacity = 128

        # Request A on locality domain-0: commit tokens
        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=0)
        self._commit_and_close(kv_a, tokens, capacity)

        # Request B on locality domain-1: must NOT find reusable blocks (salted keys differ)
        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=1)
        self.assertEqual(
            kv_b.num_committed_tokens,
            0,
            "Different locality domain must not reuse blocks from the other locality domain",
        )
        kv_b.close()

    def test_both_locality_domains_build_independent_reuse(self) -> None:
        """Each locality domain can independently build and reuse its own blocks."""
        self.manager = _make_localized_manager()
        tokens = list(range(64))
        capacity = 128

        # Request A on locality domain-0 and B on locality domain-1: both commit same tokens
        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=0)
        self._commit_and_close(kv_a, tokens, capacity)

        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=1)
        self.assertEqual(
            kv_b.num_committed_tokens, 0, "First request on locality domain-1, no reuse yet"
        )
        with TemporaryCudaStream([]) as s:
            kv_b.resume(cast(CudaStream, s.handle))
            kv_b.resize(capacity)
            kv_b.commit(tokens[kv_b.num_committed_tokens :])
            kv_b.stop_committing()
        kv_b.close()

        # Now both locality domains have committed blocks. New requests should reuse on both.
        kv_a2 = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=0)
        self.assertGreater(
            kv_a2.num_committed_tokens, 0, "locality domain-0 must reuse its own blocks"
        )
        self._commit_and_close(kv_a2, tokens, capacity)

        kv_b2 = self.manager.create_kv_cache(input_tokens=tokens[:-1], locality_domain_id=1)
        self.assertGreater(
            kv_b2.num_committed_tokens, 0, "locality domain-1 must reuse its own blocks"
        )
        self._commit_and_close(kv_b2, tokens, capacity)

    def test_non_localized_reuse_unaffected(self) -> None:
        """locality_domain_id=None must reuse blocks the same as before (no salting)."""
        self.manager = _make_non_localized_manager()
        tokens = list(range(64))
        capacity = 128

        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1])
        self._commit_and_close(kv_a, tokens, capacity)

        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1])
        self.assertGreater(
            kv_b.num_committed_tokens, 0, "Non-localized must reuse blocks as before"
        )
        self._commit_and_close(kv_b, tokens, capacity)

    def test_tree_task_id_computation(self) -> None:
        """Verify ReuseScope salting produces distinct roots for different locality_domain_ids."""
        from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import ReuseScope, RootBlock

        # Same lora, different locality_domain → different root key
        key_0 = RootBlock.make_key(ReuseScope(lora_id=42, locality_domain_id=0))
        key_1 = RootBlock.make_key(ReuseScope(lora_id=42, locality_domain_id=1))
        self.assertNotEqual(key_0, key_1)

        # Same lora, same locality_domain → same root key
        key_0b = RootBlock.make_key(ReuseScope(lora_id=42, locality_domain_id=0))
        self.assertEqual(key_0, key_0b)

        # locality_domain_id=None → unsalted lora-only namespace
        key_none = RootBlock.make_key(ReuseScope(lora_id=42))
        key_salted = RootBlock.make_key(ReuseScope(lora_id=42, locality_domain_id=0))
        self.assertNotEqual(key_none, key_salted)

        # lora=None, locality_domain=0 vs lora=None, locality_domain=1 → different
        key_no_lora_0 = RootBlock.make_key(ReuseScope(locality_domain_id=0))
        key_no_lora_1 = RootBlock.make_key(ReuseScope(locality_domain_id=1))
        self.assertNotEqual(key_no_lora_0, key_no_lora_1)

        # lora=None, locality_domain=None → default namespace
        key_both_none = RootBlock.make_key(ReuseScope())
        key_both_none_b = RootBlock.make_key(ReuseScope())
        self.assertEqual(key_both_none, key_both_none_b)


if __name__ == "__main__":
    unittest.main()
