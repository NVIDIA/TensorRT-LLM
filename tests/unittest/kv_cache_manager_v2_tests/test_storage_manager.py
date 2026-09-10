# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from typing import TYPE_CHECKING

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManagerConfig,
        LayerId,
    )
    from kv_cache_manager_v2._common import GPU_LEVEL, PRIORITY_DEFAULT, CacheTier, PageStatus
    from kv_cache_manager_v2._cuda_virt_mem import LOCALIZATION_OFFSET
    from kv_cache_manager_v2._eviction_controller import PerLevelEvictionController
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleId, LifeCycleRegistry
    from kv_cache_manager_v2._storage._config import create_storage_config
    from kv_cache_manager_v2._storage._core import (
        GpuCacheLevelStorage,
        PoolGroupIndex,
        SlotAllocator,
        SlotId,
    )
    from kv_cache_manager_v2._storage_manager import StorageManager
    from kv_cache_manager_v2._utils import exact_div, init_cuda_once, typed_range
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManagerConfig,
        LayerId,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
        GPU_LEVEL,
        PRIORITY_DEFAULT,
        CacheTier,
        PageStatus,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._cuda_virt_mem import LOCALIZATION_OFFSET
    from tensorrt_llm.runtime.kv_cache_manager_v2._eviction_controller import (
        PerLevelEvictionController,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
        LifeCycleId,
        LifeCycleRegistry,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._config import create_storage_config
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import (
        GpuCacheLevelStorage,
        PoolGroupIndex,
        SlotAllocator,
        SlotId,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import StorageManager
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        exact_div,
        init_cuda_once,
        typed_range,
    )

# Canonical role names matching FakeEngine conventions
ROLE_KEY = DataRole("key")
ROLE_VALUE = DataRole("value")
ROLE_KEY_BLOCK_QUANT = DataRole("key_block_quant")
ROLE_VALUE_BLOCK_QUANT = DataRole("value_block_quant")

# 256 MiB — small enough for CI, large enough that phys_mem_size stays at 2 MiB.
_256MB = 256 * (1 << 20)
_DEFAULT_TOKENS_PER_BLOCK = 32
_DEFAULT_KV_BUF_SIZE = 8192  # 8KB


def _make_config(
    num_layers: int,
    gpu_quota: int = _256MB,
    host_quota: int = 0,
    tokens_per_block: int = _DEFAULT_TOKENS_PER_BLOCK,
    kv_buf_size: int = _DEFAULT_KV_BUF_SIZE,
    block_quant_buf_size: int | None = None,
    swa_on_even_layers: int | None = None,
    enable_locality_domains: bool = False,
) -> KVCacheManagerConfig:
    """Build a KVCacheManagerConfig.

    swa_on_even_layers: if not None, even-indexed layers get this sliding window size;
                        odd-indexed layers have no sliding window (full attention).
    """
    cache_tiers = [
        GpuCacheTierConfig(quota=gpu_quota, enable_locality_domains=enable_locality_domains)
    ]
    if host_quota > 0:
        cache_tiers.append(HostCacheTierConfig(quota=host_quota))

    layers = []
    for i in range(num_layers):
        buffers = [
            BufferConfig(role=ROLE_KEY, size=kv_buf_size),
            BufferConfig(role=ROLE_VALUE, size=kv_buf_size),
        ]
        if block_quant_buf_size is not None:
            buffers += [
                BufferConfig(role=ROLE_KEY_BLOCK_QUANT, size=block_quant_buf_size),
                BufferConfig(role=ROLE_VALUE_BLOCK_QUANT, size=block_quant_buf_size),
            ]
        sliding_window_size = None
        if swa_on_even_layers is not None and i % 2 == 0:
            sliding_window_size = swa_on_even_layers
        layers.append(
            AttentionLayerConfig(
                layer_id=LayerId(i),
                buffers=buffers,
                sliding_window_size=sliding_window_size,
            )
        )
    return KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        cache_tiers=cache_tiers,
        layers=layers,
    )


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


def _make_storage_manager(
    config: KVCacheManagerConfig, *, localized: bool = False
) -> StorageManager:
    life_cycles = LifeCycleRegistry(config)
    storage_config = create_storage_config(config)
    with _override_localization_mode("1" if localized else "0"):
        return StorageManager(
            life_cycles,
            storage_config,
            config.tokens_per_block,
            config.swa_scratch_reuse,
        )


def _get_localized_slot_id_offset(
    sm: StorageManager, pg: PoolGroupIndex = PoolGroupIndex(0)
) -> int:
    gpu_storage = sm._levels[GPU_LEVEL].storage
    return gpu_storage._pool_groups[pg]._slot_id_offset


class _FakeEvictablePage:
    def __init__(self, life_cycle: LifeCycleId, locality_domain_id: int | None) -> None:
        self.life_cycle = life_cycle
        self.locality_domain_id = locality_domain_id
        self.cache_level = GPU_LEVEL
        self.priority = PRIORITY_DEFAULT
        self.status = PageStatus.DROPPABLE
        self.node_ref = None

    def is_committed(self) -> bool:
        return True


def setUpModule() -> None:
    # Every test here builds a Python StorageManager; under the C++ backend the
    # config classes are the C++ ones and the two cannot be mixed.
    if os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower() != "python":
        raise unittest.SkipTest("requires the Python KVCacheManagerV2 backend")


class TestStorageManagerInit(unittest.TestCase):
    """Tests for StorageManager instantiation and post-init invariants."""

    storage_manager: StorageManager | None

    def setUp(self) -> None:
        init_cuda_once()
        self.storage_manager = None

    def tearDown(self) -> None:
        if self.storage_manager is not None:
            self.storage_manager.destroy()
            self.storage_manager = None

    def test_locality_domain_disabled_by_default_when_localization_supported(self) -> None:
        """Default GPU tier config stays non-localized even on locality domain-capable hardware."""
        config = _make_config(num_layers=2)
        self.storage_manager = _make_storage_manager(config, localized=True)
        gpu_storage = self.storage_manager._levels[GPU_LEVEL].storage

        self.assertIsInstance(gpu_storage, GpuCacheLevelStorage)
        self.assertEqual(gpu_storage.num_locality_domains, 1)

    # ------------------------------------------------------------------
    # Test 1: single life cycle — all layers share the same pool group
    # ------------------------------------------------------------------
    def test_single_life_cycle_gpu_only(self) -> None:
        """4 uniform layers (no SWA) → 1 life cycle.

        K and V buffers from all 4 layers share the same size and life cycle,
        so they are coalesced into a single CoalescedBuffer → 1 pool group, 1 pool.
        """
        config = _make_config(num_layers=4)
        self.storage_manager = _make_storage_manager(config)
        sm = self.storage_manager

        # Structural invariants
        self.assertEqual(sm.num_life_cycles, 1, "Expected exactly 1 life cycle")
        self.assertEqual(sm.num_pool_groups, 1, "Expected exactly 1 pool group")
        self.assertEqual(sm.num_cache_levels, 1)
        self.assertEqual(sm.cache_tiers, (CacheTier.GPU_MEM,))

        # Pool group 0 has exactly 1 pool (all same-size buffers coalesced)
        pg_idx = PoolGroupIndex(0)
        slot_sizes = sm.slot_size(pg_idx)
        self.assertEqual(len(slot_sizes), 1, "Expected 1 pool (all same-size buffers coalesced)")

        # Each slot holds K+V for all 4 layers: 4 * 2 * kv_buf_size
        expected_slot_size = 4 * 2 * _DEFAULT_KV_BUF_SIZE  # 4 layers * 2 roles * 8KB = 64KB
        self.assertEqual(slot_sizes[0], expected_slot_size)

        # All slots must be free right after construction
        total_slots = sm.num_slots(pg_idx)
        self.assertGreater(total_slots, 0, "Pool must have at least 1 slot")
        gpu_storage = sm._levels[GPU_LEVEL].storage
        self.assertEqual(
            gpu_storage.get_num_free_slots(pg_idx),
            total_slots,
            "All slots must be free after construction",
        )

    # ------------------------------------------------------------------
    # Test 2: two life cycles, same slot sizes → merged into 1 pool group
    # ------------------------------------------------------------------
    def test_two_life_cycles_same_slot_size_merged(self) -> None:
        """4 layers: even layers SWA=100, odd layers full-attention.

        Both life cycles have identical K+V buffer sizes → same slot_size_list
        → merged into a single pool group with 2 variants.
        """
        config = _make_config(num_layers=4, swa_on_even_layers=100)
        self.storage_manager = _make_storage_manager(config)
        sm = self.storage_manager

        self.assertEqual(sm.num_life_cycles, 2, "Expected 2 distinct life cycles")
        self.assertEqual(
            sm.num_pool_groups,
            1,
            "Same slot_size_list → life cycles must be merged into 1 pool group",
        )

        # Both life cycles map to pool group 0
        for lc in typed_range(sm.num_life_cycles):
            self.assertEqual(sm.get_pool_group_index(LifeCycleId(lc)), PoolGroupIndex(0))

        # Each life cycle has 2 layers; 2 layers * 2 roles = 4 buffers → slot_size = 4 * kv_buf_size
        pg_idx = PoolGroupIndex(0)
        slot_sizes = sm.slot_size(pg_idx)
        self.assertEqual(len(slot_sizes), 1)
        expected_slot_size = 2 * 2 * _DEFAULT_KV_BUF_SIZE  # 2 layers * 2 roles * 8KB = 32KB
        self.assertEqual(slot_sizes[0], expected_slot_size)

        # All slots free at init
        total_slots = sm.num_slots(pg_idx)
        self.assertGreater(total_slots, 0)
        self.assertEqual(sm._levels[GPU_LEVEL].storage.get_num_free_slots(pg_idx), total_slots)

    # ------------------------------------------------------------------
    # Test 3: two life cycles, different slot sizes → 2 separate pool groups
    # ------------------------------------------------------------------
    def test_two_life_cycles_different_slot_sizes_separate_pool_groups(self) -> None:
        """2 layers have different buffer sizes per life cycle.

          layer 0 (no SWA): K=V=8192
          layer 1 (SWA=100): K=V=16384
        Different slot_size_lists → 2 pool groups, 1 variant each.
        """
        kv_small = 8192
        kv_large = 16384
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[
                    BufferConfig(role=ROLE_KEY, size=kv_small),
                    BufferConfig(role=ROLE_VALUE, size=kv_small),
                ],
                sliding_window_size=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[
                    BufferConfig(role=ROLE_KEY, size=kv_large),
                    BufferConfig(role=ROLE_VALUE, size=kv_large),
                ],
                sliding_window_size=100,
            ),
        ]
        config = KVCacheManagerConfig(
            tokens_per_block=_DEFAULT_TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=_256MB)],
            layers=layers,
        )
        self.storage_manager = _make_storage_manager(config)
        sm = self.storage_manager

        self.assertEqual(sm.num_life_cycles, 2)
        self.assertEqual(
            sm.num_pool_groups,
            2,
            "Different slot_size_lists → must produce 2 separate pool groups",
        )

        # Each pool group should have 1 pool (each life cycle has 1 unique buffer size)
        for pg_idx in typed_range(sm.num_pool_groups):
            slot_sizes = sm.slot_size(PoolGroupIndex(pg_idx))
            self.assertEqual(len(slot_sizes), 1)

        # Collect slot sizes from both pool groups and verify both appear
        expected_slot_size_small = kv_small * 2  # 1 layer * 2 roles * 8KB = 16KB
        expected_slot_size_large = kv_large * 2  # 1 layer * 2 roles * 16KB = 32KB
        all_slot_sizes = set()
        for pg_idx in typed_range(sm.num_pool_groups):
            sizes = sm.slot_size(PoolGroupIndex(pg_idx))
            all_slot_sizes.add(sizes[0])
        self.assertIn(expected_slot_size_small, all_slot_sizes)
        self.assertIn(expected_slot_size_large, all_slot_sizes)

        # All slots free
        for pg_idx in typed_range(sm.num_pool_groups):
            pg = PoolGroupIndex(pg_idx)
            self.assertEqual(
                sm._levels[GPU_LEVEL].storage.get_num_free_slots(pg),
                sm.num_slots(pg),
            )

    # ------------------------------------------------------------------
    # Test 4: block quantization → 2 pools in the same pool group
    # ------------------------------------------------------------------
    def test_block_quant_yields_two_pools_per_pool_group(self) -> None:
        """2 layers, no SWA, K=V=8192 + block-quant=512.

        Two distinct buffer sizes within the same life cycle → 1 pool group with 2 pools.
        Pools are ordered descending by coalesced size.
        """
        bq_size = 512
        config = _make_config(num_layers=2, block_quant_buf_size=bq_size)
        self.storage_manager = _make_storage_manager(config)
        sm = self.storage_manager

        self.assertEqual(sm.num_life_cycles, 1)
        self.assertEqual(sm.num_pool_groups, 1)

        pg_idx = PoolGroupIndex(0)
        slot_sizes = sm.slot_size(pg_idx)
        self.assertEqual(len(slot_sizes), 2, "Expected 2 pools (KV data + block quant)")

        # Pool 0 (larger): 2 layers * 2 KV roles * kv_buf_size
        # Pool 1 (smaller): 2 layers * 2 BQ roles * bq_size
        # create_storage_config sorts CoalescedBuffers descending by size.
        expected_kv_slot_size = 2 * 2 * _DEFAULT_KV_BUF_SIZE  # 2 layers * 2 roles * 8KB = 32KB
        expected_bq_slot_size = 2 * 2 * bq_size  # 2 layers * 2 roles * 512 = 2048
        self.assertEqual(slot_sizes[0], expected_kv_slot_size)
        self.assertEqual(slot_sizes[1], expected_bq_slot_size)

        # All slots free
        total = sm.num_slots(pg_idx)
        self.assertGreater(total, 0)
        self.assertEqual(sm._levels[GPU_LEVEL].storage.get_num_free_slots(pg_idx), total)

    # ------------------------------------------------------------------
    # Test 5: GPU + host tiers → 2 cache levels
    # ------------------------------------------------------------------
    def test_gpu_and_host_tiers(self) -> None:
        """Constructing a StorageManager with GPU + host tiers results in 2 cache levels.

        one per tier, with GPU as level 0.
        """
        config = _make_config(num_layers=2, gpu_quota=_256MB, host_quota=_256MB)
        self.storage_manager = _make_storage_manager(config)
        sm = self.storage_manager

        self.assertEqual(sm.num_cache_levels, 2)
        self.assertEqual(sm.cache_tiers, (CacheTier.GPU_MEM, CacheTier.HOST_MEM))

        # Both tiers must have the same pool group structure
        self.assertEqual(sm.num_pool_groups, 1)
        for level in typed_range(sm.num_cache_levels):
            level_storage = sm._levels[level].storage
            self.assertEqual(level_storage.num_pool_groups, 1)

    # ------------------------------------------------------------------
    # Test 6: destroy is idempotent
    # ------------------------------------------------------------------
    def test_destroy_is_idempotent(self) -> None:
        """StorageManager.destroy() must be safe to call more than once."""
        config = _make_config(num_layers=2)
        sm = _make_storage_manager(config)
        sm.destroy()
        sm.destroy()  # must not raise


class TestSlotAllocatorOffset(unittest.TestCase):
    """Tests for SlotAllocator with slot_id_offset.

    Motivation: SlotAllocator was redesigned to replace DynamicBitset with
    set[SlotId] and to support a slot_id_offset parameter. These tests use
    LOCALIZATION_OFFSET as a large arbitrary offset value to exercise the
    generic offset bookkeeping in isolation:
      default: slot_ids ∈ [0, capacity)
      offset:  slot_ids ∈ [LOCALIZATION_OFFSET, LOCALIZATION_OFFSET + capacity)

    SlotAllocator is pure Python, so these tests do not assert how production
    localized storage chooses its canonical slot-id stride.
    """

    def setUp(self) -> None:
        init_cuda_once()

    # ------------------------------------------------------------------
    # Test 1: without offset, slot_ids count up from 0
    # ------------------------------------------------------------------
    def test_default_slot_ids_start_at_zero(self) -> None:
        """Without slot_id_offset the first allocated slot has slot_id == 0."""
        alloc = SlotAllocator(capacity=4)
        slot = alloc.allocate()
        self.assertEqual(int(slot.slot_id), 0)
        alloc.release(slot)

    # ------------------------------------------------------------------
    # Test 2: locality_domain1 offset — slot_ids count up from LOCALIZATION_OFFSET
    # ------------------------------------------------------------------
    def test_locality_domain1_slot_ids_start_at_localization_offset(self) -> None:
        """With slot_id_offset=LOCALIZATION_OFFSET the first slot_id equals LOCALIZATION_OFFSET."""
        alloc = SlotAllocator(capacity=4, slot_id_offset=LOCALIZATION_OFFSET)
        slot = alloc.allocate()
        self.assertEqual(int(slot.slot_id), LOCALIZATION_OFFSET)
        alloc.release(slot)

    def test_consecutive_locality_domain1_slot_ids_increment_from_offset(self) -> None:
        """Consecutive locality_domain1 slot_ids are LOCALIZATION_OFFSET, +1, +2, …."""
        alloc = SlotAllocator(capacity=4, slot_id_offset=LOCALIZATION_OFFSET)
        slots = [alloc.allocate() for _ in range(3)]
        ids = [int(s.slot_id) for s in slots]
        self.assertEqual(
            ids,
            [LOCALIZATION_OFFSET, LOCALIZATION_OFFSET + 1, LOCALIZATION_OFFSET + 2],
        )
        for s in slots:
            alloc.release(s)

    # ------------------------------------------------------------------
    # Test 3: _local_idx strips the offset correctly
    # ------------------------------------------------------------------
    def test_local_idx_strips_offset(self) -> None:
        """_local_idx(slot_id) returns the 0-based per-pool index for both locality_domain0 and locality_domain1."""
        alloc0 = SlotAllocator(capacity=4)
        alloc1 = SlotAllocator(capacity=4, slot_id_offset=LOCALIZATION_OFFSET)

        s0 = alloc0.allocate()  # slot_id == 0
        s1 = alloc1.allocate()  # slot_id == LOCALIZATION_OFFSET

        self.assertEqual(alloc0._local_idx(s0.slot_id), 0)
        self.assertEqual(alloc1._local_idx(s1.slot_id), 0)

        s0b = alloc0.allocate()  # slot_id == 1
        s1b = alloc1.allocate()  # slot_id == LOCALIZATION_OFFSET + 1
        self.assertEqual(alloc0._local_idx(s0b.slot_id), 1)
        self.assertEqual(alloc1._local_idx(s1b.slot_id), 1)

        alloc0.release(s0)
        alloc0.release(s0b)
        alloc1.release(s1)
        alloc1.release(s1b)

    # ------------------------------------------------------------------
    # Test 4: _occupied_slot_ids tracks allocations and releases
    # ------------------------------------------------------------------
    def test_occupied_slot_ids_tracks_alloc_and_release(self) -> None:
        """slot_id is in _occupied_slot_ids while held; removed on release."""
        alloc = SlotAllocator(capacity=4, slot_id_offset=LOCALIZATION_OFFSET)
        slot = alloc.allocate()
        self.assertIn(slot.slot_id, alloc._occupied_slot_ids)
        expected_id = SlotId(LOCALIZATION_OFFSET)
        alloc.release(slot)
        self.assertNotIn(expected_id, alloc._occupied_slot_ids)

    # ------------------------------------------------------------------
    # Test 5: recycled slot preserves its original (offset) slot_id
    # ------------------------------------------------------------------
    def test_recycled_slot_preserves_offset_slot_id(self) -> None:
        """Releasing and re-allocating returns the same offset-encoded slot_id."""
        alloc = SlotAllocator(capacity=4, slot_id_offset=LOCALIZATION_OFFSET)
        slot = alloc.allocate()
        original_id = slot.slot_id
        alloc.release(slot)
        recycled = alloc.allocate()
        self.assertEqual(recycled.slot_id, original_id)
        alloc.release(recycled)


class TestStorageManagerLocalized(unittest.TestCase):
    """Tests for StorageManager when GpuCacheLevelStorage has num_locality_domains > 1.

    We enable locality domain in the GPU tier config and set
    TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1 to exercise the localized code paths:

    CacheLevelManager._create_cache_level_storage
      → GpuCacheLevelStorage(localized=True)
        → GpuPoolGroup (one SlotAllocator per locality domain, locality_domain1 offset by a canonical slot stride)

    Tests cover:
    - Correct storage type instantiation (num_locality_domains > 1) when locality domain is enabled and supported
    - Assertion guards: locality_domain_id must not be None for localized allocation operations
      (new_slots, new_slots_for_pool_group)
    - Slot id encoding: locality_domain0 slots below the localized slot-id offset, locality_domain1 slots at or above it
    - Slot.locality_domain_id is stamped correctly by the allocator
    - num_slots returns the per-locality domain count and both locality domains receive equal quota
    - release_slot uses slot.locality_domain_id to dispatch to the correct locality domain allocator
    """

    storage_manager: StorageManager | None

    def setUp(self) -> None:
        init_cuda_once()
        self.storage_manager = None

    def tearDown(self) -> None:
        if self.storage_manager is not None:
            self.storage_manager.destroy()
            self.storage_manager = None

    def _make_localized_sm(self, **kwargs) -> StorageManager:
        """Build a StorageManager whose GPU level uses GpuCacheLevelStorage with localization enabled."""
        kwargs.setdefault("enable_locality_domains", True)
        config = _make_config(**kwargs)
        return _make_storage_manager(config, localized=True)

    # ------------------------------------------------------------------
    # Test 1: correct storage type is selected
    # ------------------------------------------------------------------
    def test_localized_storage_type_created_when_supported(self) -> None:
        """CacheLevelManager must create num_locality_domains > 1 when locality domain is enabled and supported."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        gpu_storage = self.storage_manager._levels[GPU_LEVEL].storage
        self.assertIsInstance(gpu_storage, GpuCacheLevelStorage)
        self.assertGreater(gpu_storage.num_locality_domains, 1)

    # ------------------------------------------------------------------
    # Test 2: assertion guards — locality_domain_id required at GPU level
    # ------------------------------------------------------------------
    def test_new_slots_asserts_without_locality_domain_id(self) -> None:
        """new_slots at GPU level must raise AssertionError when locality_domain_id is omitted."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        # [1] = 1 slot requested for the single life cycle
        with self.assertRaises(AssertionError):
            sm.new_slots(GPU_LEVEL, [1])

    def test_new_slots_for_pool_group_asserts_without_locality_domain_id(self) -> None:
        """new_slots_for_pool_group at GPU level must raise AssertionError when locality_domain_id is omitted."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        with self.assertRaises(AssertionError):
            sm.new_slots_for_pool_group(GPU_LEVEL, PoolGroupIndex(0), 1)

    def test_num_slots_asserts_without_locality_domain_id(self) -> None:
        """num_slots at GPU level must raise AssertionError when locality_domain_id is omitted."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        with self.assertRaises(AssertionError):
            self.storage_manager.num_slots(PoolGroupIndex(0))

    # ------------------------------------------------------------------
    # Test 3: both locality domains receive equal slot counts (symmetric quota split)
    # ------------------------------------------------------------------
    def test_num_slots_per_locality_domain_are_equal(self) -> None:
        """GpuCacheLevelStorage splits quota evenly; both locality domains get the same slot count."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        n0 = sm.num_slots(pg, GPU_LEVEL, locality_domain_id=0)
        n1 = sm.num_slots(pg, GPU_LEVEL, locality_domain_id=1)
        self.assertGreater(n0, 0)
        self.assertEqual(n0, n1, "Both locality domains must receive the same quota")

    def test_localized_total_quota_respects_global_budget(self) -> None:
        """Localized storage must split the configured GPU quota across locality domains."""
        self.storage_manager = self._make_localized_sm(num_layers=2, gpu_quota=_256MB)
        gpu_storage = self.storage_manager._levels[GPU_LEVEL].storage
        self.assertLessEqual(gpu_storage.total_quota, _256MB)

    # ------------------------------------------------------------------
    # Test 4: slot_id encoding — locality_domain0 below localized slot offset, locality_domain1 at or above it
    # ------------------------------------------------------------------
    def test_locality_domain0_slot_ids_below_localization_offset(self) -> None:
        """Slots allocated from locality_domain0 must have slot_id below the localized slot offset."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        storage = sm._levels[GPU_LEVEL].storage
        localized_offset = _get_localized_slot_id_offset(sm, pg)
        slots = sm.new_slots_for_pool_group(GPU_LEVEL, pg, 2, locality_domain_id=0)
        try:
            for slot in slots:
                self.assertLess(int(slot.slot_id), localized_offset)
                self.assertEqual(slot.locality_domain_id, 0)
        finally:
            for slot in slots:
                storage.release(pg, slot, slot.locality_domain_id)

    def test_locality_domain1_slot_ids_at_or_above_localization_offset(self) -> None:
        """Slots allocated from locality_domain1 must have slot_id at or above the localized slot offset."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        storage = sm._levels[GPU_LEVEL].storage
        localized_offset = _get_localized_slot_id_offset(sm, pg)
        slots = sm.new_slots_for_pool_group(GPU_LEVEL, pg, 2, locality_domain_id=1)
        try:
            for slot in slots:
                self.assertGreaterEqual(int(slot.slot_id), localized_offset)
                self.assertEqual(slot.locality_domain_id, 1)
        finally:
            for slot in slots:
                storage.release(pg, slot, slot.locality_domain_id)

    def test_localized_page_index_offset_is_int32_safe(self) -> None:
        """Localized page indices must stay within the int32 host buffer contract."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        localized_slot_offset = _get_localized_slot_id_offset(sm, pg)
        n1 = sm.num_slots(pg, GPU_LEVEL, locality_domain_id=1)
        max_slot_id = SlotId(localized_slot_offset + n1 - 1)
        max_page_index = int(sm.get_page_indices_for_slot(LifeCycleId(0), max_slot_id))
        self.assertLess(max_page_index, 2**31)

    def test_localized_page_index_matches_slot_address(self) -> None:
        """Shared-pool-pointer execution must encode the full locality domain displacement in page indices."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        gpu_storage = sm._levels[GPU_LEVEL].storage
        slot = sm.new_slots_for_pool_group(GPU_LEVEL, pg, 1, locality_domain_id=1)[0]
        try:
            layer_id = LayerId(0)
            role = DataRole("key")
            attr = sm.get_buffer_attr(layer_id, role)
            pg_idx = sm.get_pool_group_index(attr.life_cycle_id)
            pool = int(sm.get_mem_pool_base_address(pg_idx, attr.pool_index))
            stride = exact_div(attr.size, attr.expansion)
            page_index = int(sm.get_page_indices_for_slot(LifeCycleId(0), slot.slot_id))
            slot_addr = int(sm.slot_address(GPU_LEVEL, pg, slot.slot_id, attr.pool_index))
            self.assertEqual(pool + stride * page_index, slot_addr)
        finally:
            gpu_storage.release(pg, slot, slot.locality_domain_id)

    # ------------------------------------------------------------------
    # Test 5: release_slot uses slot.locality_domain_id (dispatches to the correct allocator)
    # ------------------------------------------------------------------
    def test_release_slot_restores_free_count(self) -> None:
        """Allocating a slot on locality_domain1 then releasing it must restore free count.

        This verifies that release_slot forwards slot.locality_domain_id to the localized
        storage's release method.
        """
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        free_before = sm._levels[GPU_LEVEL].storage.get_num_free_slots(pg, 1)
        slots = sm.new_slots_for_pool_group(GPU_LEVEL, pg, 1, locality_domain_id=1)
        self.assertEqual(
            sm._levels[GPU_LEVEL].storage.get_num_free_slots(pg, 1),
            free_before - 1,
        )
        sm.release_slot(LifeCycleId(0), GPU_LEVEL, slots[0])
        self.assertEqual(
            sm._levels[GPU_LEVEL].storage.get_num_free_slots(pg, 1),
            free_before,
            "Free count must be restored after release_slot",
        )

    def test_eviction_filter_selects_target_locality_domain(self) -> None:
        """Localized eviction must only free pages from the locality domain being allocated."""
        ctrl = PerLevelEvictionController([PoolGroupIndex(0)], GPU_LEVEL)
        page0 = _FakeEvictablePage(LifeCycleId(0), locality_domain_id=0)
        page1 = _FakeEvictablePage(LifeCycleId(0), locality_domain_id=1)
        ctrl.schedule_for_eviction(page0)
        ctrl.schedule_for_eviction(page1)

        evicted = ctrl.evict([1], lambda page: page.locality_domain_id == 1)

        self.assertEqual(evicted[PoolGroupIndex(0)], [page1])
        self.assertIsNone(page1.node_ref)
        self.assertIsNotNone(page0.node_ref)
        self.assertEqual(
            ctrl.num_evictable_pages(PoolGroupIndex(0), lambda page: page.locality_domain_id == 0),
            1,
        )


if __name__ == "__main__":
    unittest.main()
