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
from unittest.mock import patch

import cuda.bindings.driver as drv
import numpy as np

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
    from kv_cache_manager_v2._cuda_virt_mem import PooledPhysMemAllocator
    from kv_cache_manager_v2._eviction_controller import PerLevelEvictionController
    from kv_cache_manager_v2._exceptions import LogicError
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleId, LifeCycleRegistry
    from kv_cache_manager_v2._storage._config import create_storage_config
    from kv_cache_manager_v2._storage._core import (
        GpuCacheLevelStorage,
        GpuPoolGroup,
        GpuSlotPool,
        PoolGroupIndex,
        SlotAllocator,
        _SlotIdMap,
    )
    from kv_cache_manager_v2._storage_manager import StorageManager
    from kv_cache_manager_v2._utils import _unwrap, exact_div, init_cuda_once, typed_range
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
    from tensorrt_llm.runtime.kv_cache_manager_v2._cuda_virt_mem import PooledPhysMemAllocator
    from tensorrt_llm.runtime.kv_cache_manager_v2._eviction_controller import (
        PerLevelEvictionController,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import LogicError
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
        LifeCycleId,
        LifeCycleRegistry,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._config import create_storage_config
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import (
        GpuCacheLevelStorage,
        GpuPoolGroup,
        GpuSlotPool,
        PoolGroupIndex,
        SlotAllocator,
        _SlotIdMap,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import StorageManager
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        _unwrap,
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


class TestSlotAllocatorMetadata(unittest.TestCase):
    def _allocator(self, *args, **kwargs):
        allocator = SlotAllocator(*args, **kwargs)

        def destroy():
            allocator.prepare_for_shrink(0)
            allocator.finish_shrink()

        self.addCleanup(destroy)
        return allocator

    def test_default_slot_ids_start_at_zero(self) -> None:
        alloc = self._allocator(capacity=4)
        slot = alloc.allocate()
        self.assertEqual(slot.slot_id, 0)
        self.assertIsNone(slot.locality_domain_id)
        alloc.release(slot)

    def test_domain_one_can_own_slot_zero(self) -> None:
        mapping = _SlotIdMap(4)
        mapping.append(0)
        alloc = self._allocator(4, mapping, locality_domain_id=1)
        slot = alloc.allocate()
        self.assertEqual(slot.slot_id, 0)
        self.assertEqual(slot.locality_domain_id, 1)
        alloc.release(slot)
        recycled = alloc.allocate()
        self.assertEqual(recycled.slot_id, 0)
        self.assertEqual(recycled.locality_domain_id, 1)
        alloc.release(recycled)

    def test_shrink_tracks_interleaved_ranges(self) -> None:
        mapping = _SlotIdMap(4)
        mapping.append(2)
        mapping.append(0)
        alloc = self._allocator(8, mapping, locality_domain_id=1)
        slots = alloc.allocate_multiple(8)
        self.assertEqual([s.slot_id for s in slots], [8, 9, 10, 11, 0, 1, 2, 3])
        alloc.prepare_for_shrink(4)
        self.assertEqual(set(alloc.get_slots_blocking_shrink()), {0, 1, 2, 3})
        for slot in slots[4:]:
            alloc.release(slot)
        self.assertTrue(alloc.finish_shrink())
        self.assertEqual(alloc.num_slots, 4)
        for slot in slots[:4]:
            alloc.release(slot)

    def test_wrong_domain_release_preserves_slot(self) -> None:
        mapping = _SlotIdMap(4)
        mapping.append(0)
        alloc = self._allocator(4, mapping, locality_domain_id=1)
        slot = alloc.allocate()
        with patch.object(slot, "locality_domain_id", 0):
            with self.assertRaisesRegex(LogicError, "not occupied"):
                alloc.release(slot)
            self.assertTrue(slot.has_valid_slot)
        alloc.release(slot)


class TestGpuSlotMetadata(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()

    @contextmanager
    def _group(self, counts, sizes=(256 << 10,)):
        with _override_localization_mode("1"):
            physical = PooledPhysMemAllocator.create_localized(2 << 20)
            group = GpuPoolGroup(counts, list(sizes), physical)
            try:
                yield group
            finally:
                group.destroy()
                physical.clear()

    def _assert_data(self, group, slots) -> None:
        for slot, value in slots:
            for pool in group._pools:
                address = pool.slot_address(slot.slot_id)
                self.assertEqual(address, pool._vm.address + slot.slot_id * pool.slot_size)
                data = np.zeros(16, dtype=np.uint8)
                _unwrap(drv.cuMemcpyDtoH(data.ctypes.data, address, data.nbytes))
                self.assertTrue(np.all(data == value))

    def _fill(self, group, slot, value) -> None:
        for pool in group._pools:
            _unwrap(drv.cuMemsetD8(pool.slot_address(slot.slot_id), value, pool.slot_size))

    def test_ld1_slot_zero_and_heterogeneous_pool_addressing(self) -> None:
        # Non-power-of-two scale-buffer size exercises common alignment.
        with self._group([0, 3], (4096, 192)) as group:
            slots = [(group.allocate(1), 37)]
            try:
                self.assertEqual(slots[0][0].slot_id, 0)
                self.assertEqual(slots[0][0].locality_domain_id, 1)
                group.resize_pools(3, 0)
                group._slot_allocators[0].expand(3)
                slots.append((group.allocate(0), 81))
                for slot, value in slots:
                    self._fill(group, slot, value)
                self._assert_data(group, slots)
                for pool in group._pools:
                    self.assertLess(pool._vm.virtual_bytes(), 1 << 40)
            finally:
                for slot, _ in slots:
                    group.release(slot)

    def test_grow_shrink_and_reuse_ranges_without_moving_other_domain(self) -> None:
        with self._group([16, 8]) as group:
            slots = [(group.allocate(0), 23), (group.allocate(1), 71)]
            try:
                addresses = [group.slot_address(slot.slot_id) for slot, _ in slots]
                for slot, value in slots:
                    self._fill(group, slot, value)
                allocator = group._slot_allocators[0]
                allocator.prepare_for_shrink(4)
                allocator.finish_shrink()
                group.resize_pools(4, 0)
                group.resize_pools(16, 1)
                group._slot_allocators[1].expand(16)
                extra = group.allocate_multiple(15, 1)
                slots.extend((slot, 97) for slot in extra)
                for slot in extra:
                    self._fill(group, slot, 97)
                self.assertEqual(
                    [group.slot_address(slot.slot_id) for slot, _ in slots[:2]], addresses
                )
                self.assertEqual(len({slot.slot_id for slot, _ in slots}), len(slots))
                # The recycled range precedes this domain's original range.
                self.assertLess(extra[-1].slot_id, slots[1][0].slot_id)
                self._assert_data(group, slots)
            finally:
                for slot, _ in slots:
                    group.release(slot)

    def test_multi_pool_growth_failure_rolls_back(self) -> None:
        with self._group([4, 4], (256 << 10, 128 << 10)) as group:
            original_resize = GpuSlotPool.resize
            old_ranges = [list(mapping.chunks) for mapping in group._slot_maps]
            old_bytes = [pool.num_bytes(1) for pool in group._pools]
            slot = group.allocate(1)
            self._fill(group, slot, 52)

            def fail_second_pool(pool, new_num_slots, locality_domain_id=0):
                if pool is group._pools[1] and new_num_slots > 4:
                    raise RuntimeError("injected mapping failure")
                original_resize(pool, new_num_slots, locality_domain_id)

            try:
                with patch.object(GpuSlotPool, "resize", fail_second_pool):
                    with self.assertRaisesRegex(RuntimeError, "injected mapping failure"):
                        group.resize_pools(40, 1)
                self.assertEqual([list(mapping.chunks) for mapping in group._slot_maps], old_ranges)
                self.assertEqual([pool.num_bytes(1) for pool in group._pools], old_bytes)
                self._assert_data(group, [(slot, 52)])
                # The failed reservation remains reusable by another domain.
                group.resize_pools(40, 0)
                group._slot_allocators[0].expand(40)
                self._assert_data(group, [(slot, 52)])
            finally:
                group.release(slot)

    def test_mapped_ranges_exclude_retired_storage(self) -> None:
        with self._group([16, 8]) as group:
            allocator = group._slot_allocators[0]
            allocator.prepare_for_shrink(0)
            allocator.finish_shrink()
            group.resize_pools(0, 0)
            self.assertEqual(group.slot_ranges(), [(16, 24)])
            self.assertEqual(group.slot_index_upper_bound, 24)
            pool = group._pools[0]
            self.assertEqual(pool._vm.mapped_ranges(), [(16 * pool.slot_size, 24 * pool.slot_size)])

    def test_cuda_mapping_failure_preserves_existing_pages(self) -> None:
        for operation in ("cuMemMap", "cuMemSetAccess"):
            with self.subTest(operation=operation), self._group([8, 8]) as group:
                slot = group.allocate(1)
                self._fill(group, slot, 61)
                pool = group._pools[0]
                original_ranges = pool._vm.mapped_ranges()
                original_operation = getattr(drv, operation)
                calls = 0

                def fail_second_mapping(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("injected CUDA mapping failure")
                    return original_operation(*args)

                try:
                    with patch.object(drv, operation, fail_second_mapping):
                        with self.assertRaisesRegex(RuntimeError, "injected CUDA mapping failure"):
                            group.resize_pools(32, 1)
                    self.assertEqual(pool._vm.mapped_ranges(), original_ranges)
                    self.assertEqual(group.num_slots(1), 8)
                    self._assert_data(group, [(slot, 61)])
                    group.resize_pools(32, 1)
                    group._slot_allocators[1].expand(32)
                    self._assert_data(group, [(slot, 61)])
                finally:
                    group.release(slot)


class TestStorageManagerLocalized(unittest.TestCase):
    """Tests for StorageManager when GpuCacheLevelStorage has num_locality_domains > 1.

    We enable locality domain in the GPU tier config and set
    TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1 to exercise the localized code paths:

    CacheLevelManager._create_cache_level_storage
      → GpuCacheLevelStorage(localized=True)
        → GpuPoolGroup (one SlotAllocator per locality domain, shared slot IDs with per-domain metadata)

    Tests cover:
    - Correct storage type instantiation (num_locality_domains > 1) when locality domain is enabled and supported
    - Assertion guards: locality_domain_id must not be None for localized allocation operations
      (new_slots, new_slots_for_pool_group)
    - Global slot IDs use the same address calculation in every locality domain
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
        with self.assertRaisesRegex(AssertionError, "locality_domain_id"):
            sm.new_slots(GPU_LEVEL, [1])

    def test_new_slots_for_pool_group_asserts_without_locality_domain_id(self) -> None:
        """new_slots_for_pool_group at GPU level must raise AssertionError when locality_domain_id is omitted."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        with self.assertRaisesRegex(AssertionError, "locality_domain_id"):
            sm.new_slots_for_pool_group(GPU_LEVEL, PoolGroupIndex(0), 1)

    def test_num_slots_asserts_without_locality_domain_id(self) -> None:
        """num_slots at GPU level must raise AssertionError when locality_domain_id is omitted."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        with self.assertRaisesRegex(AssertionError, "locality_domain_id"):
            self.storage_manager.num_slots(PoolGroupIndex(0))

    # ------------------------------------------------------------------
    # Test 2b: a domain id must not be forwarded to non-localized storage
    # ------------------------------------------------------------------
    def test_new_slots_for_pool_group_spills_to_host_with_locality_domain_id(self) -> None:
        """Spilling a domain-pinned cache to host must not forward the domain id.

        Host and disk storage inherit the base signature, which takes no
        locality_domain_id, so forwarding it raises TypeError.
        """
        self.storage_manager = self._make_localized_sm(num_layers=2, host_quota=_256MB)
        sm = self.storage_manager
        host_level = GPU_LEVEL + 1
        self.assertGreater(sm.num_cache_levels, host_level, "test needs a host tier")
        self.assertNotIsInstance(
            sm._levels[host_level].storage,
            GpuCacheLevelStorage,
            "host tier must not be GPU storage",
        )

        slots = sm.new_slots_for_pool_group(host_level, PoolGroupIndex(0), 1, locality_domain_id=1)
        self.assertEqual(len(slots), 1)
        self.assertIsNone(slots[0].locality_domain_id)
        sm.release_slot(LifeCycleId(0), host_level, slots[0])

    def test_new_slots_spills_to_host_with_locality_domain_id(self) -> None:
        """Same for the multi-life-cycle new_slots() entry point."""
        self.storage_manager = self._make_localized_sm(num_layers=2, host_quota=_256MB)
        sm = self.storage_manager
        host_level = GPU_LEVEL + 1
        num_slots = [0] * sm.num_life_cycles
        num_slots[LifeCycleId(0)] = 1

        ret = sm.new_slots(host_level, num_slots, locality_domain_id=1)
        self.assertEqual(len(ret[LifeCycleId(0)]), 1)
        sm.release_slot(LifeCycleId(0), host_level, ret[LifeCycleId(0)][0])

    # ------------------------------------------------------------------
    # Test 2c: resizing must keep every locality domain the same size
    # ------------------------------------------------------------------
    def _gpu_pool_group(self, sm: StorageManager, pg: PoolGroupIndex):
        return sm._levels[GPU_LEVEL].storage._pool_groups[pg]

    def test_expand_pool_group_targets_the_requested_locality_domain(self) -> None:
        """Expand must act on the named locality domain, leaving the others alone."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        pool_group = self._gpu_pool_group(sm, pg)
        self.assertGreater(pool_group.num_locality_domains, 1)

        before = [pool_group.num_slots(uid) for uid in range(pool_group.num_locality_domains)]
        # expand allocates new physical memory, which re-enters the localized
        # allocator, so it needs the mock capability the fixture only applies
        # during construction.
        with _override_localization_mode("1"):
            sm.expand_pool_group(GPU_LEVEL, pg, before[1] + 1, locality_domain_id=1)
        after = [pool_group.num_slots(uid) for uid in range(pool_group.num_locality_domains)]

        self.assertEqual(after[1], before[1] + 1, "requested locality domain must grow")
        self.assertEqual(after[0], before[0], "other locality domains must be untouched")

    def test_shrink_pool_group_targets_the_requested_locality_domain(self) -> None:
        """Shrink must act on the named locality domain, leaving the others alone."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        pool_group = self._gpu_pool_group(sm, pg)
        self.assertGreater(pool_group.num_locality_domains, 1)

        before = [pool_group.num_slots(uid) for uid in range(pool_group.num_locality_domains)]
        self.assertGreater(before[1], 1, "need room to shrink")
        sm.shrink_pool_group(GPU_LEVEL, pg, before[1] - 1, [], locality_domain_id=1)
        after = [pool_group.num_slots(uid) for uid in range(pool_group.num_locality_domains)]

        self.assertEqual(after[1], before[1] - 1, "requested locality domain must shrink")
        self.assertEqual(after[0], before[0], "other locality domains must be untouched")

    def test_resizing_each_locality_domain_keeps_them_equal(self) -> None:
        """The caller resizes every locality domain, so the halves stay symmetric."""
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        pool_group = self._gpu_pool_group(sm, pg)
        num_locality_domains = pool_group.num_locality_domains
        target = pool_group.num_slots(0) - 1

        for uid in range(num_locality_domains):
            sm.shrink_pool_group(GPU_LEVEL, pg, target, [], locality_domain_id=uid)

        sizes = [pool_group.num_slots(uid) for uid in range(num_locality_domains)]
        self.assertTrue(
            all(n == target for n in sizes),
            f"every locality domain must reach the target size: {sizes}",
        )

    def test_compute_slot_count_list_is_per_locality_domain(self) -> None:
        """The aggregate quota must be divided per locality domain, as __init__ does.

        slot_count_list reports per-locality domain counts, so a count computed from
        total_quota must be comparable with it; otherwise adjust_cache_level compares
        a total against a per-domain value and picks shrink vs expand wrongly.
        """
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        gpu_storage = sm._levels[GPU_LEVEL].storage
        num_locality_domains = gpu_storage.num_locality_domains
        self.assertGreater(num_locality_domains, 1, "test needs localized storage")

        per_locality_domain = gpu_storage.slot_count_list
        ratio_list = [1.0 / int(sm.num_pool_groups)] * int(sm.num_pool_groups)
        min_slots = sm._min_slots_for_level(GPU_LEVEL)
        computed = gpu_storage.compute_slot_count_list(
            ratio_list, min_slots, gpu_storage.total_quota
        )

        for pg in typed_range(sm.num_pool_groups):
            self.assertLessEqual(
                computed[pg],
                per_locality_domain[pg] + 1,
                f"computed count {computed[pg]} must be per-locality domain, "
                f"not the {num_locality_domains}x aggregate "
                f"(per-domain is {per_locality_domain[pg]})",
            )

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
        self.assertGreater(gpu_storage.total_quota, 0)
        # A budget that is merely under the cap would also be satisfied by
        # allocating nothing, or by giving one domain everything.
        pg = PoolGroupIndex(0)
        per_domain = [
            self.storage_manager.num_slots(pg, GPU_LEVEL, locality_domain_id=d)
            for d in range(gpu_storage.num_locality_domains)
        ]
        self.assertTrue(all(n > 0 for n in per_domain), per_domain)
        self.assertEqual(len(set(per_domain)), 1, per_domain)

    # ------------------------------------------------------------------
    # Test 4: shared slot-ID space and explicit locality metadata
    # ------------------------------------------------------------------
    def test_slot_ids_are_unique_across_domains(self) -> None:
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        slots = []
        try:
            for uid in (1, 0):
                allocated = sm.new_slots_for_pool_group(GPU_LEVEL, pg, 2, locality_domain_id=uid)
                slots.extend(allocated)
                self.assertTrue(all(slot.locality_domain_id == uid for slot in allocated))
            self.assertEqual(len({slot.slot_id for slot in slots}), len(slots))
            for slot in slots:
                self.assertEqual(
                    sm._base_page_index_for_slot(LifeCycleId(0), slot.slot_id), slot.slot_id
                )
                self.assertLess(sm.get_page_indices_for_slot(LifeCycleId(0), slot.slot_id), 2**31)
        finally:
            for slot in slots:
                sm.release_slot(LifeCycleId(0), GPU_LEVEL, slot)

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

    def test_batched_migrate_destination_stays_in_the_requested_locality_domain(self) -> None:
        """Defragmentation must draw destination slots from the source's locality domain.

        The source side already passes ``src.locality_domain_id``; the destination side
        defaulted to locality domain 0, so a domain-1 page was copied into domain 0's
        memory while its owner still addressed it as domain 1.
        """
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        pool_group = self._gpu_pool_group(sm, pg)
        self.assertGreater(pool_group.num_locality_domains, 1)

        src_slots = sm.new_slots_for_pool_group(GPU_LEVEL, pg, 1, locality_domain_id=1)
        src = _FakeEvictablePage(LifeCycleId(0), locality_domain_id=1)
        src.slot_id = src_slots[0].slot_id
        src.ready_event = src_slots[0].ready_event
        self.assertEqual(src.locality_domain_id, 1)

        dst_slots = sm._batched_migrate(
            pg,
            GPU_LEVEL,
            GPU_LEVEL,
            [src],
            update_src=False,
            defrag=True,
            dst_locality_domain_id=1,
        )
        try:
            self.assertIsNotNone(dst_slots)
            self.assertEqual(len(dst_slots), 1)
            self.assertEqual(
                dst_slots[0].locality_domain_id,
                1,
                "destination slot must come from the source's locality domain",
            )
        finally:
            for slot in dst_slots or []:
                pool_group.release(slot, 1)
            sm.release_slot(LifeCycleId(0), GPU_LEVEL, src_slots[0])

    def test_failed_migration_returns_slots_to_their_owner(self) -> None:
        self.storage_manager = self._make_localized_sm(num_layers=2)
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        group = self._gpu_pool_group(sm, pg)
        slot = group.allocate(1)
        src = _FakeEvictablePage(LifeCycleId(0), locality_domain_id=1)
        src.slot_id = slot.slot_id
        src.ready_event = slot.ready_event
        free_before = [group.num_free_slots(uid) for uid in (0, 1)]
        try:
            with patch(
                f"{StorageManager.__module__}.batched_copy",
                side_effect=RuntimeError("injected copy failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected copy failure"):
                    sm._batched_migrate(
                        pg,
                        GPU_LEVEL,
                        GPU_LEVEL,
                        [src],
                        update_src=False,
                        defrag=True,
                        dst_locality_domain_id=1,
                    )
            self.assertEqual([group.num_free_slots(uid) for uid in (0, 1)], free_before)
            self.assertTrue(slot.has_valid_slot)
            self.assertEqual(slot.locality_domain_id, 1)
        finally:
            group.release(slot)

    def test_ld1_host_roundtrip_preserves_data_and_metadata(self) -> None:
        self.storage_manager = self._make_localized_sm(
            num_layers=2, host_quota=_256MB, block_quant_buf_size=192
        )
        sm = self.storage_manager
        pg = PoolGroupIndex(0)
        lc = LifeCycleId(0)
        host_level = GPU_LEVEL + 1
        group = self._gpu_pool_group(sm, pg)
        slots = [(GPU_LEVEL, group.allocate(1))]

        def page_for(slot):
            page = _FakeEvictablePage(lc, slot.locality_domain_id)
            page.slot_id = slot.slot_id
            page.ready_event = slot.ready_event
            return page

        try:
            for i, pool in enumerate(group._pools):
                _unwrap(
                    drv.cuMemsetD8(pool.slot_address(slots[0][1].slot_id), 51 + i, pool.slot_size)
                )
            host_slots = sm._batched_migrate(
                pg, host_level, GPU_LEVEL, [page_for(slots[0][1])], update_src=False
            )
            self.assertIsNotNone(host_slots)
            slots.append((host_level, host_slots[0]))
            self.assertIsNone(host_slots[0].locality_domain_id)
            gpu_slots = sm._batched_migrate(
                pg,
                GPU_LEVEL,
                host_level,
                [page_for(host_slots[0])],
                update_src=False,
                dst_locality_domain_id=1,
            )
            self.assertIsNotNone(gpu_slots)
            slots.append((GPU_LEVEL, gpu_slots[0]))
            self.assertEqual(gpu_slots[0].locality_domain_id, 1)
            gpu_slots[0].ready_event.synchronize()
            for i, pool in enumerate(group._pools):
                data = np.zeros(pool.slot_size, dtype=np.uint8)
                _unwrap(
                    drv.cuMemcpyDtoH(
                        data.ctypes.data, pool.slot_address(gpu_slots[0].slot_id), data.nbytes
                    )
                )
                self.assertTrue(np.all(data == 51 + i))
        finally:
            for level, slot in slots:
                sm.release_slot(lc, level, slot)

    def test_eviction_filter_selects_target_locality_domain(self) -> None:
        """Localized eviction must only free pages from the locality domain being allocated."""
        ctrl = PerLevelEvictionController([PoolGroupIndex(0)], GPU_LEVEL)
        page0 = _FakeEvictablePage(LifeCycleId(0), locality_domain_id=0)
        page1 = _FakeEvictablePage(LifeCycleId(0), locality_domain_id=1)
        ctrl.schedule_for_eviction(page0)
        ctrl.schedule_for_eviction(page1)

        evicted = ctrl.evict([1], lambda page: page.locality_domain_id == 1)

        try:
            self.assertEqual(evicted[PoolGroupIndex(0)], [page1])
            self.assertIsNone(page1.node_ref)
            self.assertIsNotNone(page0.node_ref)
            self.assertEqual(
                ctrl.num_evictable_pages(
                    PoolGroupIndex(0), lambda page: page.locality_domain_id == 0
                ),
                1,
            )
        finally:
            # Under TLLM_DEBUG_MODE=1 the controller's __del__ terminates the
            # process if any eviction policy is still holding pages.
            ctrl.evict([1], lambda page: page.locality_domain_id == 0)


if __name__ == "__main__":
    unittest.main()
