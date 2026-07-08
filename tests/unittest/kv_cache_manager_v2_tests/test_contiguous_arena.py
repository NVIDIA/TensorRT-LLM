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
"""Tests for the contiguous primary KV cache storage layer.

Covers (see ``contiguous_primary_kvcache/DESIGN.md``): the block-range
allocator, int31 index-width check, page budget, sparse ``VirtMem`` mapping,
the ``SequenceArena`` demand-paging layer, and the ``ArenaPoolGroup`` /
``GpuArenaCacheLevelStorage`` storage seam.

The allocator / index-width / budget / config tests are pure logic and always
run. The sparse ``VirtMem``, ``SequenceArena``, and pool-group tests require a
CUDA device.
"""

import unittest
from importlib.util import find_spec
from typing import TYPE_CHECKING

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    _CUDA_AVAILABLE = False

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        ContiguousArenaConfig,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        WriteThroughPolicy,
        rawref,
    )
    from kv_cache_manager_v2._common import GPU_LEVEL, CacheLevel, LayerId, Priority
    from kv_cache_manager_v2._copy_engine import CopyTask
    from kv_cache_manager_v2._cuda_virt_mem import PooledPhysMemAllocator, VirtMem
    from kv_cache_manager_v2._exceptions import LogicError, OutOfPagesError
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleId, LifeCycleRegistry
    from kv_cache_manager_v2._page import Page
    from kv_cache_manager_v2._sequence_arena import (
        INT31_MAX,
        BlockRangeAllocator,
        PageBudget,
        SequenceArena,
        check_index_width,
    )
    from kv_cache_manager_v2._storage._config import create_storage_config
    from kv_cache_manager_v2._storage._core import (
        ArenaPoolGroup,
        GpuArenaCacheLevelStorage,
        PoolGroupIndex,
        PoolIndex,
    )
    from kv_cache_manager_v2._storage_manager import StorageManager, _coalesce_copy_tasks
    from kv_cache_manager_v2._utils import CachedCudaEvent, TemporaryCudaStream, init_cuda_once
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        ContiguousArenaConfig,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        WriteThroughPolicy,
        rawref,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
        GPU_LEVEL,
        CacheLevel,
        LayerId,
        Priority,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import CopyTask
    from tensorrt_llm.runtime.kv_cache_manager_v2._cuda_virt_mem import (
        PooledPhysMemAllocator,
        VirtMem,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import LogicError, OutOfPagesError
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
        LifeCycleId,
        LifeCycleRegistry,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._page import Page
    from tensorrt_llm.runtime.kv_cache_manager_v2._sequence_arena import (
        INT31_MAX,
        BlockRangeAllocator,
        PageBudget,
        SequenceArena,
        check_index_width,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._config import create_storage_config
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import (
        ArenaPoolGroup,
        GpuArenaCacheLevelStorage,
        PoolGroupIndex,
        PoolIndex,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import (
        StorageManager,
        _coalesce_copy_tasks,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        CachedCudaEvent,
        TemporaryCudaStream,
        init_cuda_once,
    )

MiB = 1 << 20
requires_cuda = unittest.skipUnless(_CUDA_AVAILABLE, "requires CUDA")


class TestBlockRangeAllocator(unittest.TestCase):
    def test_alignment_derivation(self) -> None:
        # record_stride divides page -> align = page / stride blocks
        a = BlockRangeAllocator(capacity_blocks=64, record_stride=1 * MiB, phys_page_size=2 * MiB)
        self.assertEqual(a.align_blocks, 2)
        # record_stride multiple of page -> each block is whole pages -> align 1
        a = BlockRangeAllocator(capacity_blocks=64, record_stride=4 * MiB, phys_page_size=2 * MiB)
        self.assertEqual(a.align_blocks, 1)
        # coprime-ish stride -> align = page / gcd
        a = BlockRangeAllocator(capacity_blocks=64, record_stride=3 * MiB, phys_page_size=2 * MiB)
        self.assertEqual(a.align_blocks, 2)  # gcd(3,2)=1 -> 2MiB/1MiB=2

    def test_allocate_is_page_aligned_and_padded(self) -> None:
        a = BlockRangeAllocator(capacity_blocks=64, record_stride=1 * MiB, phys_page_size=2 * MiB)
        # request 3 -> padded up to align(2)=4
        base = a.allocate(3)
        self.assertEqual(base % a.align_blocks, 0)
        self.assertEqual(a.reserved_len(base), 4)

    def test_no_two_ranges_share_a_page(self) -> None:
        # stride 1MiB, page 2MiB: two blocks per page. Allocations must not let
        # two ranges land in the same physical page.
        a = BlockRangeAllocator(capacity_blocks=64, record_stride=1 * MiB, phys_page_size=2 * MiB)
        bases = [a.allocate(1) for _ in range(4)]
        # every base is a multiple of align (2), so byte offset is a page multiple
        for b in bases:
            self.assertEqual((b * MiB) % (2 * MiB), 0)
        self.assertEqual(len(set(bases)), len(bases))

    def test_free_and_coalesce(self) -> None:
        a = BlockRangeAllocator(capacity_blocks=16, record_stride=2 * MiB, phys_page_size=2 * MiB)
        b0 = a.allocate(4)
        b1 = a.allocate(4)
        b2 = a.allocate(4)
        self.assertEqual(a.free_blocks(), 4)
        # free the middle then the ends; everything should coalesce back to 16
        a.free(b1)
        a.free(b0)
        a.free(b2)
        self.assertEqual(a.free_blocks(), 16)
        self.assertEqual(a.largest_free_blocks(), 16)
        # can allocate the whole arena again
        self.assertEqual(a.allocate(16), 0)

    def test_reuse_after_free(self) -> None:
        a = BlockRangeAllocator(capacity_blocks=8, record_stride=2 * MiB, phys_page_size=2 * MiB)
        b = a.allocate(8)
        a.free(b)
        self.assertEqual(a.allocate(8), b)

    def test_out_of_space_raises(self) -> None:
        a = BlockRangeAllocator(capacity_blocks=8, record_stride=2 * MiB, phys_page_size=2 * MiB)
        a.allocate(8)
        with self.assertRaises(MemoryError):
            a.allocate(1)

    def test_double_free_guard(self) -> None:
        a = BlockRangeAllocator(capacity_blocks=8, record_stride=2 * MiB, phys_page_size=2 * MiB)
        b = a.allocate(4)
        a.free(b)
        with self.assertRaises(KeyError):
            a.free(b)  # base no longer live


class TestIndexWidthCheck(unittest.TestCase):
    def test_within_limit_ok(self) -> None:
        # 8k sequences * 4k blocks with scale 1 is far below the ceiling
        check_index_width(block_capacity=1 << 20, num_coalesced_subbuffers=32)

    def test_exactly_at_limit_ok(self) -> None:
        # capacity*scale - 1 == INT31_MAX must pass
        scale = 2
        cap = (INT31_MAX + 1) // scale
        check_index_width(block_capacity=cap, num_coalesced_subbuffers=scale)

    def test_over_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            check_index_width(block_capacity=(1 << 26), num_coalesced_subbuffers=64)


class TestContiguousArenaConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        c = ContiguousArenaConfig()
        self.assertEqual(c.phys_page_size, 2 * MiB)
        self.assertEqual(c.write_through, WriteThroughPolicy.ON_FREE)
        self.assertFalse(c.lazy_gpu_retention)

    def test_super_page_ok(self) -> None:
        ContiguousArenaConfig(phys_page_size=16 * MiB)

    def test_unaligned_page_size_raises(self) -> None:
        with self.assertRaises(AssertionError):
            ContiguousArenaConfig(phys_page_size=3 * MiB)

    def test_negative_margin_raises(self) -> None:
        with self.assertRaises(AssertionError):
            ContiguousArenaConfig(map_ahead_pages=-1)


@requires_cuda
class TestSparseVirtMem(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        self.allocator = PooledPhysMemAllocator(2 * MiB)

    def test_map_unmap_bookkeeping(self) -> None:
        vm = VirtMem(64 * MiB, self.allocator)
        try:
            vm.map_range(0, 4)
            self.assertEqual(vm.num_sparse_chunks, 4)
            self.assertTrue(vm.is_mapped(0))
            self.assertTrue(vm.is_mapped(3 * 2 * MiB))
            self.assertFalse(vm.is_mapped(4 * 2 * MiB))
            # map a non-adjacent run at a higher offset
            vm.map_range(8 * 2 * MiB, 2)
            self.assertEqual(vm.num_sparse_chunks, 6)
            vm.unmap_range(0, 4)
            self.assertEqual(vm.num_sparse_chunks, 2)
            vm.unmap_range(8 * 2 * MiB, 2)
            self.assertEqual(vm.num_sparse_chunks, 0)
        finally:
            vm.destroy()

    def test_unaligned_offset_asserts(self) -> None:
        vm = VirtMem(16 * MiB, self.allocator)
        try:
            with self.assertRaises(AssertionError):
                vm.map_range(1 * MiB, 1)  # not a multiple of 2 MiB
        finally:
            vm.destroy()

    def test_mode_mutual_exclusion(self) -> None:
        vm = VirtMem(16 * MiB, self.allocator)
        try:
            vm.extend(1)  # tail-stack mode
            with self.assertRaises(AssertionError):
                vm.map_range(4 * MiB, 1)  # cannot mix
        finally:
            vm.destroy()

    def test_data_roundtrip(self) -> None:
        # Write through a torch tensor viewing the mapped VA, read it back.
        vm = VirtMem(8 * MiB, self.allocator)
        try:
            vm.map_range(0, 1)
            n = (2 * MiB) // 4
            t = self._tensor_at(int(vm.address), n)
            t.fill_(1.5)
            torch.cuda.synchronize()
            self.assertTrue(bool((t == 1.5).all().item()))
        finally:
            vm.destroy()

    def test_alias_map_shares_bytes_and_refcounts(self) -> None:
        """P3 aliasing: one physical handle mapped at two VA addresses.

        Both mappings (across two reservations) show the same bytes;
        unmapping either alias in either order keeps the handle alive
        until the last reference drops.
        """
        vm_a = VirtMem(8 * MiB, self.allocator)
        vm_b = VirtMem(8 * MiB, self.allocator)
        try:
            vm_a.map_range(0, 2)
            shared = [vm_a.shared_chunk(0), vm_a.shared_chunk(2 * MiB)]
            self.assertEqual([s.num_refs for s in shared], [1, 1])
            # alias into a DIFFERENT reservation at a different offset
            vm_b.map_alias(4 * MiB, shared)
            self.assertEqual([s.num_refs for s in shared], [2, 2])
            self.assertEqual(vm_b.num_sparse_chunks, 2)
            n = (2 * MiB) // 4
            src = self._tensor_at(int(vm_a.address), n)
            dst = self._tensor_at(int(vm_b.address) + 4 * MiB, n)
            src.fill_(3.25)
            torch.cuda.synchronize()
            self.assertTrue(bool((dst == 3.25).all().item()))
            # owner unmaps first; alias keeps the bytes (refcount holds)
            torch.cuda.synchronize()
            vm_a.unmap_range(0, 2)
            self.assertEqual([s.num_refs for s in shared], [1, 1])
            dst2 = self._tensor_at(int(vm_b.address) + 4 * MiB, n)
            self.assertTrue(bool((dst2 == 3.25).all().item()))
            torch.cuda.synchronize()
            vm_b.unmap_range(4 * MiB, 2)
            self.assertEqual([s.num_refs for s in shared], [0, 0])
        finally:
            vm_a.destroy()
            vm_b.destroy()

    def test_alias_destroy_drops_refs(self) -> None:
        """destroy() of an alias-holding reservation drops references.

        The shared handle survives while another mapping lives.
        """
        vm_a = VirtMem(4 * MiB, self.allocator)
        vm_b = VirtMem(4 * MiB, self.allocator)
        try:
            vm_a.map_range(0, 1)
            shared = vm_a.shared_chunk(0)
            vm_b.map_alias(0, [shared])
            self.assertEqual(shared.num_refs, 2)
            vm_b.destroy()
            self.assertEqual(shared.num_refs, 1)
            self.assertTrue(vm_a.is_mapped(0))
            n = (2 * MiB) // 4
            t = self._tensor_at(int(vm_a.address), n)
            t.fill_(9.5)
            torch.cuda.synchronize()
            self.assertTrue(bool((t == 9.5).all().item()))
        finally:
            vm_a.destroy()

    def test_alias_collision_asserts(self) -> None:
        vm = VirtMem(8 * MiB, self.allocator)
        try:
            vm.map_range(0, 1)
            shared = vm.shared_chunk(0)
            with self.assertRaises(AssertionError):
                vm.map_alias(0, [shared])  # chunk already mapped
        finally:
            vm.destroy()

    @staticmethod
    def _tensor_at(ptr: int, num_float32: int) -> "torch.Tensor":
        # Build a torch tensor aliasing an existing device pointer via the
        # CUDA array interface.
        class _Aliaser:
            __cuda_array_interface__ = {
                "shape": (num_float32,),
                "typestr": "<f4",
                "data": (ptr, False),
                "version": 3,
                "strides": None,
            }

        return torch.as_tensor(_Aliaser(), device="cuda")


@requires_cuda
class TestSequenceArena(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        self.allocator = PooledPhysMemAllocator(2 * MiB)

    def _arena(self, record_stride: int, capacity: int, margin: int = 1) -> "SequenceArena":
        return SequenceArena(
            block_capacity=capacity,
            record_stride=record_stride,
            phys_mem_allocator=self.allocator,
            map_ahead_pages=margin,
        )

    def test_reserve_then_demand_map(self) -> None:
        # 1 block = 2 physical pages; 16 blocks capacity.
        arena = self._arena(record_stride=4 * MiB, capacity=16, margin=1)
        try:
            base = arena.reserve(8)
            self.assertEqual(arena.mapped_pages, 0)  # reservation maps nothing
            arena.ensure_mapped(base, 3)  # 3 blocks -> 6 pages, +1 margin = 7
            self.assertEqual(arena.mapped_pages, 7)
            # growing is incremental: only new pages get mapped
            arena.ensure_mapped(base, 4)  # now up to 8 pages + margin, but 7 already mapped
            self.assertEqual(arena.mapped_pages, 9)
        finally:
            arena.destroy()

    def test_map_ahead_clamped_to_reserved(self) -> None:
        arena = self._arena(record_stride=2 * MiB, capacity=8, margin=100)
        try:
            base = arena.reserve(2)  # 2 blocks -> 2 pages reserved
            arena.ensure_mapped(base, 2)
            # margin is huge but must not exceed the 2 reserved pages
            self.assertEqual(arena.mapped_pages, 2)
        finally:
            arena.destroy()

    def test_deferred_reclaim_frees_pages_and_range(self) -> None:
        arena = self._arena(record_stride=4 * MiB, capacity=16, margin=0)
        try:
            base = arena.reserve(4)
            arena.ensure_mapped(base, 4)
            self.assertEqual(arena.mapped_pages, 8)
            # NULL event is always complete -> reclaim happens on first drain
            arena.enqueue_free(base, CachedCudaEvent.NULL)
            reclaimed = arena.drain_reclaim()
            self.assertEqual(reclaimed, 1)
            self.assertEqual(arena.mapped_pages, 0)
            # the block range is back and reusable
            self.assertEqual(arena.reserve(16), 0)
        finally:
            arena.destroy()

    def test_two_sequences_disjoint_pages(self) -> None:
        arena = self._arena(record_stride=1 * MiB, capacity=32, margin=0)
        try:
            a = arena.reserve(3)
            b = arena.reserve(3)
            self.assertNotEqual(a, b)
            arena.ensure_mapped(a, 3)
            pages_a = arena.mapped_pages
            arena.ensure_mapped(b, 3)
            # mapping b must not have touched a's already-mapped pages twice
            self.assertGreater(arena.mapped_pages, pages_a)
        finally:
            arena.destroy()

    def test_alias_prefix_shares_pages_uncharged(self) -> None:
        """P3: a second sequence alias-maps the canonical prefix's pages.

        Zero budget charge, correct frontier (growth continues past the
        alias), and reclaim releases only the charged surplus while the
        alias holds the handles alive.
        """
        budget = PageBudget(16)
        arena = SequenceArena(
            block_capacity=16,
            record_stride=4 * MiB,  # 1 block = 2 pages
            phys_mem_allocator=self.allocator,
            map_ahead_pages=0,
            page_budget=budget,
        )
        try:
            a = arena.reserve(4)
            arena.ensure_mapped(a, 4)  # 8 pages charged
            self.assertEqual(budget.used_pages, 8)
            # 3 of A's 4 blocks are the "committed prefix": fully covered
            # chunks only -> 3 blocks * 4MiB / 2MiB = 6 chunks, all full.
            self.assertEqual(arena.aliasable_prefix_blocks(a, 3), 3)
            shared = arena.shared_prefix_chunks(a, 3)
            self.assertEqual(sum(len(s) for s in shared), 6)
            b = arena.reserve(4)
            self.assertEqual(arena.alias_prefix(b, shared), 6)
            self.assertEqual(budget.used_pages, 8)  # aliases charge nothing
            self.assertEqual(arena.aliased_pages_in_range(b), 6)
            self.assertEqual(arena.mapped_pages_in_range(b), 6)
            self.assertEqual(arena.charged_pages_in_range(b), 0)
            # growth continues after the aliased frontier
            arena.ensure_mapped(b, 4)  # 8 chunks target; 6 aliased -> +2
            self.assertEqual(budget.used_pages, 10)
            self.assertEqual(arena.charged_pages_in_range(b), 2)
            # write via A's VA, read via B's VA (same physical bytes)
            n = (2 * MiB) // 4
            addr_a = int(arena.base_address(0)) + a * 4 * MiB
            addr_b = int(arena.base_address(0)) + b * 4 * MiB
            ta = TestSparseVirtMem._tensor_at(addr_a, n)
            ta.fill_(5.75)
            torch.cuda.synchronize()
            tb = TestSparseVirtMem._tensor_at(addr_b, n)
            self.assertTrue(bool((tb == 5.75).all().item()))
            # reclaim A (canonical owner): B's aliases keep the handles
            torch.cuda.synchronize()
            arena.reclaim(a)
            self.assertEqual(budget.used_pages, 2)  # only A's own 8 released
            tb2 = TestSparseVirtMem._tensor_at(addr_b, n)
            self.assertTrue(bool((tb2 == 5.75).all().item()))
            # reclaim B: releases only its charged surplus (2), aliased 6 not
            torch.cuda.synchronize()
            arena.reclaim(b)
            self.assertEqual(budget.used_pages, 0)
            self.assertEqual(arena.mapped_pages, 0)
        finally:
            arena.destroy()

    def test_aliasable_prefix_partial_boundary_page(self) -> None:
        """A half-covered boundary chunk is trimmed from the aliasable span.

        The remainder falls back to the copy path.
        """
        arena = self._arena(record_stride=1 * MiB, capacity=32, margin=0)  # 2 blocks/page
        try:
            base = arena.reserve(8)
            arena.ensure_mapped(base, 8)
            self.assertEqual(arena.aliasable_prefix_blocks(base, 3), 2)  # 3rd block half-page
            self.assertEqual(arena.aliasable_prefix_blocks(base, 4), 4)
            shared = arena.shared_prefix_chunks(base, 3)
            self.assertEqual(sum(len(s) for s in shared), 1)  # only the full chunk
        finally:
            arena.destroy()


class TestPageBudget(unittest.TestCase):
    def test_consume_release(self) -> None:
        b = PageBudget(4)
        self.assertEqual(b.free_pages, 4)
        b.consume(3)
        self.assertEqual(b.used_pages, 3)
        self.assertEqual(b.free_pages, 1)
        b.release(2)
        self.assertEqual(b.used_pages, 1)

    def test_over_consume_raises_without_side_effects(self) -> None:
        b = PageBudget(4)
        b.consume(3)
        with self.assertRaises(OutOfPagesError):
            b.consume(2)
        self.assertEqual(b.used_pages, 3)  # unchanged after the failed consume


class TestMultiPoolAlignment(unittest.TestCase):
    def test_alignment_is_lcm_across_pools(self) -> None:
        # stride 1MiB -> align 2; stride 512KiB -> align 4; lcm = 4
        a = BlockRangeAllocator(
            capacity_blocks=64, record_stride=(1 * MiB, MiB // 2), phys_page_size=2 * MiB
        )
        self.assertEqual(a.align_blocks, 4)
        # whole-page stride (align 1) with a half-page stride (align 2) -> 2
        a = BlockRangeAllocator(
            capacity_blocks=64, record_stride=(2 * MiB, 1 * MiB), phys_page_size=2 * MiB
        )
        self.assertEqual(a.align_blocks, 2)

    def test_ranges_page_disjoint_in_every_pool(self) -> None:
        strides = (1 * MiB, MiB // 2)
        a = BlockRangeAllocator(capacity_blocks=64, record_stride=strides, phys_page_size=2 * MiB)
        bases = [a.allocate(1) for _ in range(4)]
        for base in bases:
            for stride in strides:
                self.assertEqual((base * stride) % (2 * MiB), 0)


@requires_cuda
class TestSequenceArenaMultiPool(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        self.allocator = PooledPhysMemAllocator(2 * MiB)

    def test_maps_all_pools(self) -> None:
        # pool 0: 1 block = 1 page; pool 1: 2 blocks = 1 page
        arena = SequenceArena(
            block_capacity=8,
            record_stride=(2 * MiB, 1 * MiB),
            phys_mem_allocator=self.allocator,
            map_ahead_pages=0,
        )
        try:
            self.assertEqual(arena.num_pools, 2)
            base = arena.reserve(4)
            new_pages = arena.ensure_mapped(base, 4)
            # pool 0: 4 pages, pool 1: 2 pages
            self.assertEqual(new_pages, 6)
            self.assertEqual(arena.mapped_pages, 6)
            self.assertNotEqual(int(arena.base_address(0)), int(arena.base_address(1)))
        finally:
            arena.destroy()

    def test_budget_enforced_atomically(self) -> None:
        budget = PageBudget(4)
        arena = SequenceArena(
            block_capacity=8,
            record_stride=(2 * MiB, 1 * MiB),
            phys_mem_allocator=self.allocator,
            map_ahead_pages=0,
            page_budget=budget,
        )
        try:
            base = arena.reserve(4)  # full mapping would need 6 pages > 4
            with self.assertRaises(OutOfPagesError):
                arena.ensure_mapped(base, 4)
            # nothing was mapped or consumed by the failed call
            self.assertEqual(arena.mapped_pages, 0)
            self.assertEqual(budget.used_pages, 0)
            # a smaller frontier fits: 2 blocks -> 2 + 1 = 3 pages
            self.assertEqual(arena.ensure_mapped(base, 2), 3)
            self.assertEqual(budget.used_pages, 3)
            # reclaim returns the pages to the budget
            arena.enqueue_free(base, CachedCudaEvent.NULL)
            self.assertEqual(arena.drain_reclaim(), 1)
            self.assertEqual(budget.used_pages, 0)
        finally:
            arena.destroy()


@requires_cuda
class TestArenaPoolGroup(unittest.TestCase):
    PAGE = 2 * MiB

    def setUp(self) -> None:
        init_cuda_once()
        self.allocator = PooledPhysMemAllocator(self.PAGE)
        self.budget = PageBudget(64)

    def _group(self, block_capacity: int = 16, map_ahead: int = 0) -> "ArenaPoolGroup":
        return ArenaPoolGroup(
            block_capacity=block_capacity,
            slot_size_list=[2 * MiB, 1 * MiB],
            shared_phys_mem_pool=self.allocator,
            page_budget=self.budget,
            map_ahead_pages=map_ahead,
            page_index_scale=64,
        )

    def test_int31_check_wired_into_construction(self) -> None:
        with self.assertRaises(ValueError):
            ArenaPoolGroup(
                block_capacity=INT31_MAX,
                slot_size_list=[2 * MiB],
                shared_phys_mem_pool=self.allocator,
                page_budget=self.budget,
                map_ahead_pages=0,
                page_index_scale=2,
            )

    def test_sequence_slots_are_consecutive_and_addressable(self) -> None:
        group = self._group()
        try:
            rng = group.reserve_sequence(4)
            group.ensure_mapped(rng, 4)
            slots = [group.take_slot(rng, j) for j in range(4)]
            ids = [s.slot_id for s in slots]
            self.assertEqual(ids, list(range(rng.base_block, rng.base_block + 4)))
            for pool_idx, stride in enumerate((2 * MiB, 1 * MiB)):
                addrs = [group._pools[pool_idx].slot_address(i) for i in ids]
                deltas = [addrs[j + 1] - addrs[j] for j in range(len(addrs) - 1)]
                self.assertEqual(deltas, [stride] * 3)  # contiguous in VA
            for s in slots:
                group.release(s)
            group.free_sequence(rng, CachedCudaEvent.NULL)
            # The fenced drain parks the range still mapped (freed-range
            # adoption): pages stay charged but become retained/spillable.
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 1)
            self.assertEqual(group.mapped_pages, 6)
            self.assertEqual(self.budget.used_pages, 6)
            self.assertEqual(self.budget.retained_pages, 6)
            # Pressure spill actually unmaps and releases.
            self.assertEqual(group.spill_retained(1 << 62), 6)
            self.assertEqual(group.mapped_pages, 0)
            self.assertEqual(self.budget.used_pages, 0)
            self.assertEqual(self.budget.retained_pages, 0)
        finally:
            group.destroy()

    def test_reclaim_gated_on_outstanding_slots(self) -> None:
        group = self._group()
        try:
            rng = group.reserve_sequence(2)
            group.ensure_mapped(rng, 2)
            slot = group.take_slot(rng, 0)
            group.free_sequence(rng, CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 0)  # slot still outstanding
            group.release(slot)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 1)
        finally:
            group.destroy()

    def test_reclaim_gated_on_iteration_fence(self) -> None:
        """Risk #3: reclaim requires a later execution-stream fence.

        A freed range must NOT be reclaimed before an
        execution-stream fence recorded at a later drain point completes —
        the overlap scheduler may have speculatively enqueued a step that
        still reads the range when free_sequence runs.
        """
        group = self._group()
        try:
            rng = group.reserve_sequence(2)
            group.ensure_mapped(rng, 2)
            group.free_sequence(rng, CachedCudaEvent.NULL)
            # Unfenced drains (e.g. the impl's internal pressure drains) must
            # never reclaim a range that has no fence assigned yet.
            self.assertEqual(group.drain_reclaim(), 0)
            self.assertEqual(group.drain_reclaim(None), 0)
            self.assertGreater(group.mapped_pages, 0)
            # Assign-only fence, then an unfenced drain may reclaim (into the
            # parked/adoptable pool — pages stay mapped until spilled).
            group.fence_pending_frees(CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(), 1)
            self.assertGreater(group.mapped_pages, 0)
            group.spill_retained(1 << 62)
        finally:
            group.destroy()

    def test_blocking_drain_waits_for_gate_events(self) -> None:
        """§4.6 preemption headroom: the blocking drain waits out gate events.

        ``drain_reclaim(wait=True)`` blocks on a
        pending gate event (e.g. a suspend-evacuation copy still in flight)
        and processes the range in the same call (parking it for adoption).
        The scheduler's mid-pass eviction path depends on this — a
        non-waiting drain skips the range, the freed pages stay charged and
        unadoptable, and a fully suspended batch deadlocks.
        """
        group = self._group()
        try:
            rng = group.reserve_sequence(2)
            group.ensure_mapped(rng, 2)
            slot = group.take_slot(rng, 0)
            # Simulate an in-flight evacuation copy: the released slot's
            # ready event is recorded behind a device sleep.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(50_000_000)
            ev = CachedCudaEvent(CudaStream(stream.cuda_stream))
            slot.ready_event = ev
            group.release(slot)
            group.free_sequence(rng, CachedCudaEvent.NULL)
            # Pending gate: the plain drain skips the range...
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 0)
            # ...and the blocking drain waits it out and parks it (freed-range
            # adoption keeps the pages mapped; retained = spillable/adoptable).
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL, wait=True), 1)
            self.assertTrue(ev.query_complete())
            self.assertEqual(self.budget.used_pages, self.budget.retained_pages)
            group.spill_retained(1 << 62)
            self.assertEqual(group.mapped_pages, 0)
            self.assertEqual(self.budget.used_pages, 0)
        finally:
            group.destroy()

    def test_blocking_drain_respects_outstanding_slots_and_fence(self) -> None:
        """``wait=True`` must not bypass non-event gating conditions.

        It must not hang on (or reclaim) ranges whose gating
        conditions are host-side state rather than events: an outstanding
        slot has no event to wait for, and an unfenced range keeps its
        risk-#3 protection.
        """
        group = self._group()
        try:
            # Outstanding slot: skipped, promptly.
            rng = group.reserve_sequence(2)
            group.ensure_mapped(rng, 2)
            slot = group.take_slot(rng, 0)
            group.free_sequence(rng, CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL, wait=True), 0)
            group.release(slot)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL, wait=True), 1)
            # Unfenced range: wait=True with no fence must not bypass risk #3.
            rng2 = group.reserve_sequence(2)
            group.ensure_mapped(rng2, 2)
            group.free_sequence(rng2, CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(None, wait=True), 0)
            self.assertGreater(group.mapped_pages, 0)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL, wait=True), 1)
            # Both processed ranges are parked for adoption; spill releases.
            group.spill_retained(1 << 62)
            self.assertEqual(group.mapped_pages, 0)
        finally:
            group.destroy()

    def test_scattered_allocation_retired(self) -> None:
        group = self._group()
        try:
            with self.assertRaises(LogicError):
                group.allocate()
            with self.assertRaises(LogicError):
                group.allocate_multiple(2)
        finally:
            group.destroy()

    def test_va_exhaustion_raises_memory_error(self) -> None:
        group = self._group(block_capacity=8)
        try:
            rng = group.reserve_sequence(8)
            with self.assertRaises(MemoryError):
                group.reserve_sequence(1)
            group.free_sequence(rng, CachedCudaEvent.NULL)
            group.drain_reclaim(CachedCudaEvent.NULL)
        finally:
            group.destroy()

    def test_adoption_reuses_parked_range_without_new_maps(self) -> None:
        """Freed-range adoption: parked ranges are reused without new maps.

        A fenced-drained range is handed whole to
        the next reservation — same base, mapped prefix intact, zero new
        cuMemMap calls (the anti-churn fix for the driver TLB interference
        IMA — WORK_LOG 2026-07-07).
        """
        group = self._group()
        try:
            rng = group.reserve_sequence(4)
            self.assertEqual(group.ensure_mapped(rng, 4), 6)  # 4 + 2 pages
            base = rng.base_block
            group.free_sequence(rng, CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 1)
            self.assertEqual(self.budget.retained_pages, 6)
            adopted = group.reserve_sequence(4)
            self.assertEqual(adopted.base_block, base)
            self.assertEqual(self.budget.retained_pages, 0)  # charge kept, not retained
            self.assertEqual(self.budget.used_pages, 6)
            # the mapped prefix carries over: nothing new to map
            self.assertEqual(group.ensure_mapped(adopted, 4), 0)
            self.assertEqual(group.mapped_pages, 6)
            group.free_sequence(adopted, CachedCudaEvent.NULL)
            group.drain_reclaim(CachedCudaEvent.NULL)
        finally:
            group.destroy()

    def test_adoption_skips_unfit_and_unready_ranges(self) -> None:
        group = self._group()
        try:
            rng = group.reserve_sequence(2)
            group.ensure_mapped(rng, 2)
            base = rng.base_block
            group.free_sequence(rng, CachedCudaEvent.NULL)
            # not drained yet -> nothing parked -> fresh VA
            bigger = group.reserve_sequence(4)
            self.assertNotEqual(bigger.base_block, base)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 1)
            # parked, but too small for 4 blocks -> fresh VA again
            another = group.reserve_sequence(4)
            self.assertNotEqual(another.base_block, base)
            # fits a 2-block request -> adopted
            fit = group.reserve_sequence(2)
            self.assertEqual(fit.base_block, base)
            for r in (bigger, another, fit):
                group.free_sequence(r, CachedCudaEvent.NULL)
            group.drain_reclaim(CachedCudaEvent.NULL)
        finally:
            group.destroy()

    def test_adoption_disabled_restores_unmap_on_drain(self) -> None:
        group = self._group()
        group._range_adoption = False
        try:
            rng = group.reserve_sequence(4)
            group.ensure_mapped(rng, 4)
            base = rng.base_block
            group.free_sequence(rng, CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 1)
            self.assertEqual(group.mapped_pages, 0)
            self.assertEqual(self.budget.used_pages, 0)
            fresh = group.reserve_sequence(4)
            self.assertEqual(fresh.base_block, base)  # allocator reuse, unmapped
            self.assertEqual(group.ensure_mapped(fresh, 4), 6)  # must remap
            group.free_sequence(fresh, CachedCudaEvent.NULL)
            group.drain_reclaim(CachedCudaEvent.NULL)
        finally:
            group.destroy()

    class _FakeCanonicalPage:
        """Stand-in for a canonical (host-tier) page in registry tests."""

        __slots__ = ("__rawref__",)

        def __init__(self) -> None:
            self.__rawref__ = rawref.NULL

    def test_canonical_span_registry_alias_roundtrip(self) -> None:
        """P3 registry roundtrip: register, alias, park, adopt affinely.

        The aliased chunks charge nothing, parking carries the alias
        signature, and charged accounting stays exact through every
        transition.
        """
        group = self._group()
        group._prefix_aliasing = True
        try:
            # Owner A: 4 blocks; blocks 0-1 are the "committed prefix".
            # Pools (2MiB, 1MiB) on 2MiB pages: prefix covers pool0 2 full
            # chunks + pool1 1 full chunk -> span = 3 pages.
            rng_a = group.reserve_sequence(4)
            self.assertEqual(group.ensure_mapped(rng_a, 4), 6)
            self.assertEqual(self.budget.used_pages, 6)
            canon = [self._FakeCanonicalPage(), self._FakeCanonicalPage()]
            group.register_canonical_span(rng_a, canon, CachedCudaEvent.NULL)
            key = id(canon[0])
            self.assertEqual(rng_a._alias_span_key, (key, 2))
            self.assertEqual(group.canonical_span_pages, 3)
            # charge moved range -> registry (net zero), retained-at-will
            self.assertEqual(self.budget.used_pages, 6)
            self.assertEqual(self.budget.retained_pages, 3)

            # Reuse admission B: registry hit requires full-span coverage.
            self.assertIsNone(group.lookup_canonical_span(canon[0], canon[:1]))
            hit = group.lookup_canonical_span(canon[0], canon)
            self.assertIsNotNone(hit)
            self.assertEqual(hit[2], 2)  # usable extent = the verified match
            rng_b = group.reserve_sequence(4)  # nothing parked yet -> fresh
            self.assertNotEqual(rng_b.base_block, rng_a.base_block)
            self.assertEqual(group.alias_span_into_range(rng_b, hit[0], hit[1], hit[2]), 3)
            self.assertEqual(self.budget.used_pages, 6)  # aliases charge nothing
            # B's remainder maps fresh: pool0 +2, pool1 +1
            self.assertEqual(group.ensure_mapped(rng_b, 4), 3)
            self.assertEqual(self.budget.used_pages, 9)

            # Same physical bytes through both ranges' pool-0 VA addresses.
            n = (2 * MiB) // 4
            addr_a = group._pools[0].slot_address(rng_a.base_block)
            addr_b = group._pools[0].slot_address(rng_b.base_block)
            ta = TestSparseVirtMem._tensor_at(int(addr_a), n)
            ta.fill_(11.5)
            torch.cuda.synchronize()
            tb = TestSparseVirtMem._tensor_at(int(addr_b), n)
            self.assertTrue(bool((tb == 11.5).all().item()))

            # Owner A parks with its signature; only same-key admissions
            # adopt it. A None-key reservation must get fresh VA instead.
            group.free_sequence(rng_a, CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(CachedCudaEvent.NULL), 1)
            plain = group.reserve_sequence(4)  # alias_key=None
            self.assertNotEqual(plain.base_block, rng_a.base_block)
            # extent is part of the signature: a shorter-extent admission
            # must NOT adopt the owner's full-span range
            self.assertNotEqual(
                group.reserve_sequence(4, alias_key=(key, 1)).base_block, rng_a.base_block
            )
            adopted = group.reserve_sequence(4, alias_key=(key, 2))
            self.assertEqual(adopted.base_block, rng_a.base_block)
            self.assertEqual(adopted._alias_span_key, (key, 2))

            # Spill the registry: pins drop, lookups miss, aliases keep bytes.
            torch.cuda.synchronize()
            self.assertEqual(group.spill_canonical_spans(1 << 62), 3)
            self.assertIsNone(group.lookup_canonical_span(canon[0], canon))
            tb2 = TestSparseVirtMem._tensor_at(int(addr_b), n)
            self.assertTrue(bool((tb2 == 11.5).all().item()))

            for base, rng in list(group._ranges.items()):
                if not rng._freed:
                    group.free_sequence(rng, CachedCudaEvent.NULL)
            group.drain_reclaim(CachedCudaEvent.NULL)
            torch.cuda.synchronize()
            group.spill_retained(1 << 62)
            self.assertEqual(self.budget.used_pages, 0)
            self.assertEqual(group.mapped_pages, 0)
        finally:
            group.destroy()


@requires_cuda
class TestGpuArenaCacheLevelStorage(unittest.TestCase):
    PAGE = 2 * MiB

    def setUp(self) -> None:
        init_cuda_once()

    def test_total_quota_reports_physical_not_va(self) -> None:
        """``total_quota`` must be the PHYSICAL page-budget bytes.

        The base implementation sums pool byte sizes = the VA reservation
        extent; the KV memory estimator credits ``total_quota`` back as a
        "temporary pool to be freed", so reporting VA over-grants the final
        KV budget (observed +8 GiB on H100: 60.8 vs 52.5 GiB quota).
        """
        storage = GpuArenaCacheLevelStorage(
            slot_size_lists=[[2 * MiB], [1 * MiB]],
            block_capacity_list=[16, 16],
            page_index_scale_list=[64, 32],
            quota=8 * self.PAGE,
            phys_page_size=self.PAGE,
            map_ahead_pages=0,
        )
        try:
            # VA extent is 16 blocks x (2 MiB + 1 MiB) = 48 MiB >> quota.
            self.assertEqual(storage.total_quota, 8 * self.PAGE)
        finally:
            storage.destroy()

    def test_budget_shared_across_pool_groups(self) -> None:
        storage = GpuArenaCacheLevelStorage(
            slot_size_lists=[[2 * MiB], [1 * MiB]],
            block_capacity_list=[16, 16],
            page_index_scale_list=[64, 32],
            quota=8 * self.PAGE,
            phys_page_size=self.PAGE,
            map_ahead_pages=0,
        )
        try:
            self.assertEqual(storage.page_budget.total_pages, 8)
            g0 = storage.pool_group(0)
            g1 = storage.pool_group(1)
            r0 = g0.reserve_sequence(6)  # 6 pages in pool group 0
            g0.ensure_mapped(r0, 6)
            self.assertEqual(storage.page_budget.used_pages, 6)
            r1 = g1.reserve_sequence(8)  # would need 4 pages; only 2 left
            with self.assertRaises(OutOfPagesError):
                g1.ensure_mapped(r1, 8)
            g1.ensure_mapped(r1, 4)  # 2 pages fit
            self.assertEqual(storage.page_budget.used_pages, 8)
            # free the first sequence; the second can now finish mapping
            g0.free_sequence(r0, CachedCudaEvent.NULL)
            g1.free_sequence(r1, CachedCudaEvent.NULL)
            self.assertEqual(storage.drain_reclaim(CachedCudaEvent.NULL), 2)
            # both ranges parked for adoption; spill returns the pages
            self.assertEqual(storage.page_budget.used_pages, storage.page_budget.retained_pages)
            storage.spill_retained(1 << 62)
            self.assertEqual(storage.page_budget.used_pages, 0)
        finally:
            storage.destroy()

    def test_int31_violation_fails_before_cuda_setup(self) -> None:
        with self.assertRaises(ValueError):
            GpuArenaCacheLevelStorage(
                slot_size_lists=[[2 * MiB]],
                block_capacity_list=[INT31_MAX],
                page_index_scale_list=[2],
                quota=8 * self.PAGE,
                phys_page_size=self.PAGE,
                map_ahead_pages=0,
            )


class TestCoalesceCopyTasks(unittest.TestCase):
    SIZE = 1024

    def test_contiguous_run_merges(self) -> None:
        tasks = [CopyTask(1000 + i * self.SIZE, 5000 + i * self.SIZE) for i in range(4)]
        merged = _coalesce_copy_tasks(self.SIZE, tasks)
        self.assertEqual(merged, {4 * self.SIZE: [tasks[0]]})

    def test_broken_run_splits(self) -> None:
        # dst contiguous but src jumps after 2 tasks -> two runs of 2
        tasks = [
            CopyTask(1000, 5000),
            CopyTask(1000 + self.SIZE, 5000 + self.SIZE),
            CopyTask(1000 + 2 * self.SIZE, 9000),
            CopyTask(1000 + 3 * self.SIZE, 9000 + self.SIZE),
        ]
        merged = _coalesce_copy_tasks(self.SIZE, tasks)
        self.assertEqual(merged, {2 * self.SIZE: [tasks[0], tasks[2]]})

    def test_scattered_tasks_pass_through(self) -> None:
        tasks = [CopyTask(0, 8 * self.SIZE), CopyTask(4 * self.SIZE, 0)]
        merged = _coalesce_copy_tasks(self.SIZE, tasks)
        self.assertEqual(merged, {self.SIZE: tasks})


def _make_manager_config(
    gpu_quota: int, host_quota: int, tokens_per_block: int = 16, kv_buf_size: int = 8192
) -> "KVCacheManagerConfig":
    """A minimal two-tier (GPU + host) manager config.

    Two attention layers, KEY+VALUE buffers of one size -> a single pool
    group with one coalesced pool of slot size 4 * kv_buf_size and page-index
    scale 4.
    """
    return KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        vocab_size=4096,
        cache_tiers=[
            GpuCacheTierConfig(quota=gpu_quota),
            HostCacheTierConfig(quota=host_quota),
        ],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(layer_id),
                buffers=[
                    BufferConfig(role=DataRole("key"), size=kv_buf_size),
                    BufferConfig(role=DataRole("value"), size=kv_buf_size),
                ],
            )
            for layer_id in range(2)
        ],
    )


@requires_cuda
class TestStorageManagerArenaMode(unittest.TestCase):
    """The storage-manager seam (DESIGN.md §5, `_storage_manager.py` row).

    Covers arena-mode construction, retired-path guards, per-sequence
    pass-throughs, page-based utilization, and explicit-destination
    migration (§4.3/§4.4).
    """

    PAGE = 2 * MiB
    GPU_QUOTA = 32 * MiB  # 16 pages
    SLOT_SIZE = 4 * 8192  # 2 layers x (KEY+VALUE) coalesced
    PG0 = PoolGroupIndex(0)
    LC0 = LifeCycleId(0)

    def setUp(self) -> None:
        init_cuda_once()
        cfg = _make_manager_config(gpu_quota=self.GPU_QUOTA, host_quota=32 * MiB)
        self.storage = StorageManager(
            LifeCycleRegistry(cfg),
            create_storage_config(cfg),
            cfg.tokens_per_block,
            None,
            arena_config=ContiguousArenaConfig(phys_page_size=self.PAGE, map_ahead_pages=0),
        )

    def tearDown(self) -> None:
        self.storage.destroy()
        del self.storage

    def _gpu_page(self, slot) -> "Page":
        page = Page(
            None,
            CachedCudaEvent.NULL,
            rawref.ref(self.storage),
            self.LC0,
            GPU_LEVEL,
            Priority(50),
            None,
            None,
        )
        page.set_slot(slot)
        return page

    def _host_page(self, slot) -> "Page":
        page = Page(
            None,
            CachedCudaEvent.NULL,
            rawref.ref(self.storage),
            self.LC0,
            CacheLevel(1),
            Priority(50),
            None,
            None,
        )
        page.set_slot(slot)
        return page

    def _slot_floats(self, level: "CacheLevel", slot_id: int) -> "torch.Tensor":
        addr = int(self.storage.slot_address(level, self.PG0, slot_id, PoolIndex(0)))
        n = self.SLOT_SIZE // 4
        if level == GPU_LEVEL:
            return TestSparseVirtMem._tensor_at(addr, n)
        import ctypes

        buf = (ctypes.c_float * n).from_address(addr)
        return torch.frombuffer(buf, dtype=torch.float32)

    def test_construction_and_sizing(self) -> None:
        storage = self.storage
        self.assertTrue(storage.is_arena_mode)
        gpu_storage = storage._levels[GPU_LEVEL].storage
        self.assertIs(type(gpu_storage), GpuArenaCacheLevelStorage)
        self.assertEqual(storage.gpu_page_budget.total_pages, self.GPU_QUOTA // self.PAGE)
        # 4 coalesced sub-buffers, expansion 1 -> scale 4
        self.assertEqual(storage._compute_page_index_scales(), [4])
        # default VA: overcommit x (quota // record_stride)
        self.assertEqual(storage.num_slots(self.PG0), 8 * (self.GPU_QUOTA // self.SLOT_SIZE))
        # host level is a classic slot pool, unaffected
        self.assertGreater(storage.num_slots(self.PG0, CacheLevel(1)), 0)

    def test_retired_paths_raise(self) -> None:
        storage = self.storage
        with self.assertRaises(LogicError):
            storage.new_gpu_slots([1])
        with self.assertRaises(LogicError):
            storage.new_slots_for_pool_group(GPU_LEVEL, self.PG0, 1)
        with self.assertRaises(LogicError):
            storage.prepare_free_slots(GPU_LEVEL, [1])
        with self.assertRaises(LogicError):
            storage.adjust_cache_level(GPU_LEVEL, None, [1.0])
        # host-level allocation still works
        slots = storage.new_slots(CacheLevel(1), [2])
        for s in slots[self.LC0]:
            storage.release_slot(self.LC0, CacheLevel(1), s)

    def test_sequence_lifecycle_and_utilization(self) -> None:
        storage = self.storage
        self.assertEqual(storage.get_utilization(GPU_LEVEL), [0.0])
        rng = storage.reserve_gpu_sequence(self.PG0, 64)
        mapped = storage.ensure_gpu_mapped(self.PG0, rng, 64)  # 64 x 32 KiB = 1 page
        self.assertEqual(mapped, 1)
        budget = storage.gpu_page_budget
        self.assertEqual(budget.used_pages, 1)
        self.assertEqual(storage.get_utilization(GPU_LEVEL), [1 / budget.total_pages])
        self.assertEqual(storage.get_overall_utilization(GPU_LEVEL), 1 / budget.total_pages)
        slot = storage.take_gpu_sequence_slot(self.PG0, rng, 0)
        self.assertEqual(slot.slot_id, rng.base_block)
        storage.release_slot(self.LC0, GPU_LEVEL, slot)
        storage.free_gpu_sequence(self.PG0, rng, CachedCudaEvent.NULL)
        self.assertEqual(storage.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        self.assertEqual(budget.used_pages, budget.retained_pages)  # parked
        storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)

    def test_offload_migrates_data_and_gates_reclaim(self) -> None:
        """§4.3 active->stale mechanics.

        Explicit GPU->host migration of arena slots moves the bytes,
        retargets the pages, and returns the arena slots to their range with
        the copy event gating reclaim.
        """
        storage = self.storage
        rng = storage.reserve_gpu_sequence(self.PG0, 64)
        storage.ensure_gpu_mapped(self.PG0, rng, 2)
        pages = []
        for ordinal, value in ((0, 1.25), (1, -3.0)):
            slot = storage.take_gpu_sequence_slot(self.PG0, rng, ordinal)
            self._slot_floats(GPU_LEVEL, slot.slot_id).fill_(value)
            pages.append(self._gpu_page(slot))
        torch.cuda.synchronize()

        storage._batched_migrate(self.PG0, CacheLevel(1), GPU_LEVEL, pages, update_src=True)
        torch.cuda.synchronize()

        for page, value in zip(pages, (1.25, -3.0)):
            self.assertEqual(page.cache_level, CacheLevel(1))
            host_data = self._slot_floats(CacheLevel(1), page.slot_id)
            self.assertTrue(bool((host_data == value).all().item()))
        # the vacated arena slots returned to the range: reclaim succeeds
        storage.free_gpu_sequence(self.PG0, rng, CachedCudaEvent.NULL)
        self.assertEqual(storage.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        self.assertEqual(storage.gpu_page_budget.used_pages, storage.gpu_page_budget.retained_pages)
        storage.spill_gpu_retained(1 << 62)
        self.assertEqual(storage.gpu_page_budget.used_pages, 0)
        for page in pages:
            storage.release_slot(self.LC0, CacheLevel(1), page)

    def test_batched_migrate_orders_after_extra_prior_event(self) -> None:
        """§4.5 write-out ordering, mechanism level.

        The batched-copy kernel reads its sources at execution time with no
        ordering against other streams. A suspend/close write-out's source
        blocks may still be receiving the in-flight step's KV writes on the
        forward stream (overlap scheduler); ``extra_prior_event`` must gate
        the copy or the host tier captures the block pre-write.
        """
        storage = self.storage
        rng = storage.reserve_gpu_sequence(self.PG0, 64)
        storage.ensure_gpu_mapped(self.PG0, rng, 2)
        # Warm up the batched-copy kernel first: its first-call module-load
        # latency exceeds the race window and would mask the assertion.
        s0 = storage.take_gpu_sequence_slot(self.PG0, rng, 0)
        p0 = self._gpu_page(s0)
        storage._batched_migrate(self.PG0, CacheLevel(1), GPU_LEVEL, [p0], update_src=True)
        torch.cuda.synchronize()
        # The race: a delayed write on a foreign ("forward") stream is still
        # pending when the write-out is enqueued.
        s1 = storage.take_gpu_sequence_slot(self.PG0, rng, 1)
        sid1 = s1.slot_id
        self._slot_floats(GPU_LEVEL, sid1).fill_(1.0)
        torch.cuda.synchronize()
        page = self._gpu_page(s1)
        fwd = torch.cuda.Stream()
        with torch.cuda.stream(fwd):
            torch.cuda._sleep(100_000_000)
            self._slot_floats(GPU_LEVEL, sid1).fill_(2.0)
        prior = CachedCudaEvent(CudaStream(fwd.cuda_stream))
        storage._batched_migrate(
            self.PG0, CacheLevel(1), GPU_LEVEL, [page], update_src=True, extra_prior_event=prior
        )
        torch.cuda.synchronize()
        host_data = self._slot_floats(CacheLevel(1), page.slot_id)
        self.assertTrue(bool((host_data == 2.0).all().item()))
        storage.free_gpu_sequence(self.PG0, rng, CachedCudaEvent.NULL)
        self.assertEqual(storage.drain_gpu_reclaim(CachedCudaEvent.NULL, wait=True), 1)
        for p in (p0, page):
            storage.release_slot(self.LC0, CacheLevel(1), p)

    def test_onboard_explicit_destination(self) -> None:
        """§4.4 stale->active mechanics.

        Copies host pages into explicit, consecutive arena destinations
        without touching the sources (the host copies stay valid for the
        radix tree).
        """
        storage = self.storage
        host_slots = storage.new_slots(CacheLevel(1), [2])[self.LC0]
        src_pages = []
        for slot, value in zip(host_slots, (7.5, 0.5)):
            self._slot_floats(CacheLevel(1), slot.slot_id).fill_(value)
            src_pages.append(self._host_page(slot))

        rng = storage.reserve_gpu_sequence(self.PG0, 64)
        storage.ensure_gpu_mapped(self.PG0, rng, 2)
        dst_slots = [storage.take_gpu_sequence_slot(self.PG0, rng, j) for j in range(2)]
        ret = storage._batched_migrate(
            self.PG0,
            GPU_LEVEL,
            CacheLevel(1),
            src_pages,
            update_src=False,
            dst_slots=dst_slots,
        )
        assert ret is not None
        torch.cuda.synchronize()

        for slot, value in zip(ret, (7.5, 0.5)):
            gpu_data = self._slot_floats(GPU_LEVEL, slot.slot_id)
            self.assertTrue(bool((gpu_data == value).all().item()))
        # sources untouched: still valid host pages with their data
        for page, value in zip(src_pages, (7.5, 0.5)):
            self.assertEqual(page.cache_level, CacheLevel(1))
            host_data = self._slot_floats(CacheLevel(1), page.slot_id)
            self.assertTrue(bool((host_data == value).all().item()))

        for slot in ret:
            storage.release_slot(self.LC0, GPU_LEVEL, slot)
        storage.free_gpu_sequence(self.PG0, rng, CachedCudaEvent.NULL)
        self.assertEqual(storage.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        for page in src_pages:
            storage.release_slot(self.LC0, CacheLevel(1), page)

    def test_write_through_copies_without_moving(self) -> None:
        """§4.3 write-through-on-commit mechanics.

        Host copies materialize; sources keep their GPU slots.
        """
        storage = self.storage
        rng = storage.reserve_gpu_sequence(self.PG0, 64)
        storage.ensure_gpu_mapped(self.PG0, rng, 1)
        slot = storage.take_gpu_sequence_slot(self.PG0, rng, 0)
        gpu_slot_id = slot.slot_id
        self._slot_floats(GPU_LEVEL, gpu_slot_id).fill_(42.0)
        page = self._gpu_page(slot)
        torch.cuda.synchronize()

        host_slots = storage.write_through_pages(self.PG0, [page])
        torch.cuda.synchronize()

        self.assertEqual(len(host_slots), 1)
        self.assertEqual(page.cache_level, GPU_LEVEL)  # source not moved
        self.assertEqual(page.slot_id, gpu_slot_id)
        host_data = self._slot_floats(CacheLevel(1), host_slots[0].slot_id)
        self.assertTrue(bool((host_data == 42.0).all().item()))

        storage.release_slot(self.LC0, CacheLevel(1), host_slots[0])
        storage.release_slot(self.LC0, GPU_LEVEL, page)
        storage.free_gpu_sequence(self.PG0, rng, CachedCudaEvent.NULL)
        self.assertEqual(storage.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)


@requires_cuda
class TestArenaKVCacheManagerEndToEnd(unittest.TestCase):
    """The `_KVCache` growth seam (DESIGN.md §5, `_core/_kv_cache.py` row).

    Covers create -> resume (VA reserve) -> resize growth (demand map +
    consecutive slots) -> commit -> close (copy-on-free write-out +
    deferred reclaim).
    """

    TPB = 16  # tokens per block

    def setUp(self) -> None:
        init_cuda_once()
        cfg = _make_manager_config(gpu_quota=32 * MiB, host_quota=32 * MiB)
        cfg.contiguous_arena = ContiguousArenaConfig(map_ahead_pages=0)
        self.manager = KVCacheManager(cfg)

    def tearDown(self) -> None:
        self.manager.shutdown()
        del self.manager

    def test_grow_commit_close_roundtrip(self) -> None:
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        kv = manager.create_kv_cache(max_capacity=8 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertEqual(budget.used_pages, 0)  # VA reserve maps nothing
            self.assertTrue(kv.resize(4 * self.TPB))
            self.assertGreater(budget.used_pages, 0)
            # the sequence's kernel-visible page indices are consecutive
            indices = list(kv.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, list(range(indices[0], indices[0] + 4)))
            # growth extends the same run
            self.assertTrue(kv.resize(6 * self.TPB))
            indices = list(kv.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, list(range(indices[0], indices[0] + 6)))
            kv.commit([100 + i for i in range(4 * self.TPB)])
            kv.close()
        s.take_finish_event().synchronize()
        # copy-on-free write-out finished -> the range is reclaimable
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)
        # the 4 committed blocks live on in the host tier
        host_stats = manager._storage.get_statistics(CacheLevel(1))[0]
        self.assertEqual(host_stats.total - host_stats.free, 4)

    def test_clamp_max_seq_len_counts_pages(self) -> None:
        """§4.6 capacity planning in arena mode counts physical pages.

        Not VA slots: 16 budget pages x 64 blocks/page x 16 tokens/block.
        """
        manager = self.manager
        blocks_per_page = (2 * MiB) // (4 * 8192)  # 64: 32 KiB block record
        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(1, 10**6), 16 * blocks_per_page * self.TPB
        )
        # batch 4: the 3 other single-token sequences take a page each
        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(4, 10**6), 13 * blocks_per_page * self.TPB
        )
        # a bound below capacity is returned unchanged
        self.assertEqual(manager.clamp_max_seq_len_for_mem(1, 500), 500)

    def test_max_capacity_required_and_enforced(self) -> None:
        manager = self.manager
        with self.assertRaises(ValueError):
            manager.create_kv_cache()  # arena mode requires max_capacity
        kv = manager.create_kv_cache(max_capacity=2 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(2 * self.TPB))
            with self.assertRaises(ValueError):
                kv.resize(3 * self.TPB)  # beyond the VA reservation
            kv.close()
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def test_growth_backs_off_on_page_exhaustion(self) -> None:
        manager = self.manager
        # 16 budget pages; 1 block = 32 KiB -> 64 blocks per 2 MiB page.
        kv1 = manager.create_kv_cache(max_capacity=1024 * self.TPB)
        kv2 = manager.create_kv_cache(max_capacity=512 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv2.resume(stream))
            self.assertTrue(kv1.resize(1024 * self.TPB))  # takes all 16 pages
            self.assertFalse(kv2.resize(512 * self.TPB))  # backs off, no crash
            kv1.close()
            # kv1's range reclaim is gated on its finish event AND an
            # execution-stream fence assigned at an iteration-boundary drain
            # (risk #3; the pyexecutor adapter fences every iteration). Model
            # that boundary, then kv2's growth succeeds on the freed pages.
            torch.cuda.synchronize()
            manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
            self.assertTrue(kv2.resize(512 * self.TPB))
            kv2.close()
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def _block_floats(self, page_index: int) -> "torch.Tensor":
        addr = int(
            self.manager._storage.slot_address(
                GPU_LEVEL, PoolGroupIndex(0), page_index, PoolIndex(0)
            )
        )
        return TestSparseVirtMem._tensor_at(addr, (4 * 8192) // 4)

    def test_reuse_onboarding_roundtrip(self) -> None:
        """§4.4 stale->active reuse round-trip.

        A reuse hit copies the matched prefix from the host tier into the new
        sequence's arena range as private pages; the canonical host copies
        survive the second sequence's close (no duplicates).
        """
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        tokens = [200 + i for i in range(2 * self.TPB)]
        kv1 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv1.history_length, 0)  # empty radix tree: no match
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(2 * self.TPB))
            for j, idx in enumerate(kv1.get_base_page_indices(LifeCycleId(0))):
                self._block_floats(idx).fill_(1.0 + j)
            torch.cuda.synchronize()
            kv1.commit(tokens)
            kv1.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        host_stats = manager._storage.get_statistics(CacheLevel(1))[0]
        self.assertEqual(host_stats.total - host_stats.free, 2)

        # arena mode matches full blocks only (partial matching disabled)
        kv2 = manager.create_kv_cache(
            input_tokens=tokens + [900, 901, 902], max_capacity=4 * self.TPB
        )
        self.assertEqual(kv2.history_length, 2 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv2.resume(stream))
            torch.cuda.synchronize()
            indices = list(kv2.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, list(range(indices[0], indices[0] + 2)))
            for j, idx in enumerate(indices):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 1.0 + j).all().item()))
            # growth continues the same consecutive run past the reused prefix
            self.assertTrue(kv2.resize(3 * self.TPB))
            indices = list(kv2.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, list(range(indices[0], indices[0] + 3)))
            kv2.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)
        # private copies were dropped, not offloaded: still exactly 2 host blocks
        host_stats = manager._storage.get_statistics(CacheLevel(1))[0]
        self.assertEqual(host_stats.total - host_stats.free, 2)

    def test_reuse_match_must_fit_max_capacity(self) -> None:
        manager = self.manager
        tokens = [300 + i for i in range(2 * self.TPB)]
        kv1 = manager.create_kv_cache(input_tokens=tokens, max_capacity=2 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(2 * self.TPB))
            kv1.commit(tokens)
            kv1.close()
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
        with self.assertRaises(ValueError):
            manager.create_kv_cache(input_tokens=tokens, max_capacity=self.TPB)

    def _host_used(self) -> int:
        stats = self.manager._storage.get_statistics(CacheLevel(1))[0]
        return stats.total - stats.free

    def test_resume_gate_drains_deferred_reclaim(self) -> None:
        """Anti-livelock (§4.2 x §4.6).

        Arena frees are deferred, so page utilization can stay pinned above
        max_util_for_resume after other sequences finished. resume() must
        drain the reclaim queue before evaluating the gate, or a scheduler
        retrying resume forever livelocks (observed in the KV-cache-capacity
        estimation phase).
        """
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        kv1 = manager.create_kv_cache(max_capacity=1024 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(1024 * self.TPB))  # maps the full budget
            self.assertEqual(budget.used_pages, budget.total_pages)
            kv1.close()
        s.take_finish_event().synchronize()
        # reclaim is queued but deliberately NOT drained: utilization reads 100%.
        # Assign the iteration-boundary fence only (risk #3) — in production the
        # adapter fences every iteration; resume()'s internal drain may then
        # reclaim fenced entries.
        manager.fence_gpu_reclaim(CachedCudaEvent.NULL)
        self.assertEqual(budget.used_pages, budget.total_pages)
        kv2 = manager.create_kv_cache(max_capacity=2 * self.TPB)
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv2.resume(stream2))  # must drain, not refuse
            self.assertTrue(kv2.resize(2 * self.TPB))
            kv2.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)

    def test_intra_batch_commit_conflict_keeps_committing(self) -> None:
        """§4.4 intra-batch commit conflict.

        When another sequence committed the same tokens first (shared system
        prompts under concurrency), the loser must keep its own pages as
        private committed copies and continue committing -- not assert and
        not stop (rebasing is disabled in arena mode).
        """
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        tokens = [600 + i for i in range(2 * self.TPB)]
        kv1 = manager.create_kv_cache(max_capacity=4 * self.TPB)
        kv2 = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv2.resume(stream))
            self.assertTrue(kv1.resize(2 * self.TPB))
            self.assertTrue(kv2.resize(2 * self.TPB))
            kv1.commit(tokens)  # registers both blocks in the radix tree
            kv2.commit(tokens)  # conflict: same tokens, must go private
            self.assertEqual(kv2.num_committed_tokens, 2 * self.TPB)
            self.assertEqual(int(kv2._num_committed_blocks), 2)
            kv1.close()
            kv2.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 2)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)
        # only the canonical copies were offloaded: no duplicates on host
        self.assertEqual(self._host_used(), 2)
        # and the canonical entries remain reusable
        kv3 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv3.history_length, 2 * self.TPB)
        kv3.close()

    def test_suspend_resume_roundtrip(self) -> None:
        """§4.5 suspend/resume round-trip.

        Suspend writes committed AND uncommitted blocks out to host and
        releases the VA ranges; resume reserves fresh ranges and copies the
        resident state back, byte-identical.
        """
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        kv = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(3 * self.TPB))
            for j, idx in enumerate(kv.get_base_page_indices(LifeCycleId(0))):
                self._block_floats(idx).fill_(1.0 + j)
            torch.cuda.synchronize()
            kv.commit([400 + i for i in range(2 * self.TPB)])  # block 2 stays uncommitted
            kv.suspend()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)
        self.assertEqual(self._host_used(), 3)  # 2 canonical + 1 uncommitted

        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv.resume(stream2))
            torch.cuda.synchronize()
            indices = list(kv.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, list(range(indices[0], indices[0] + 3)))
            for j, idx in enumerate(indices):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 1.0 + j).all().item()))
            # the moved-back uncommitted block freed its host slot
            self.assertEqual(self._host_used(), 2)
            # growth continues the same consecutive run
            self.assertTrue(kv.resize(4 * self.TPB))
            indices = list(kv.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, list(range(indices[0], indices[0] + 4)))
            kv.close()
        s2.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)
        self.assertEqual(self._host_used(), 2)

    def test_suspend_orders_evacuation_after_forward_stream_writes(self) -> None:
        """§4.5 write-out ordering vs the in-flight step.

        A scheduler eviction can suspend a sequence whose current step is
        still writing its tail block on the FORWARD stream; the sequence's
        manager-stream events and the pages' ready events do not order
        against it. The evacuation copies must wait on the caller-supplied
        ``prior_event`` or the host copy captures the block pre-write and
        resume restores stale bytes.
        """
        manager = self.manager
        # Warm up the batched-copy kernel: its first-call module-load latency
        # exceeds the race window below and would mask the assertion.
        kv0 = manager.create_kv_cache(max_capacity=self.TPB)
        with TemporaryCudaStream([]) as s0:
            self.assertTrue(kv0.resume(CudaStream(s0.handle)))
            self.assertTrue(kv0.resize(self.TPB))
            kv0.suspend()
        s0.take_finish_event().synchronize()
        kv0.close()
        torch.cuda.synchronize()
        kv = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(2 * self.TPB))
            old_indices = list(kv.get_base_page_indices(LifeCycleId(0)))
            for idx in old_indices:
                self._block_floats(idx).fill_(1.0)
            torch.cuda.synchronize()
            kv.commit([500 + i for i in range(self.TPB)])  # block 1 = uncommitted tail
            # The in-flight "forward step": a delayed overwrite of the tail
            # block on a separate stream, still pending when suspend() runs.
            fwd = torch.cuda.Stream()
            with torch.cuda.stream(fwd):
                torch.cuda._sleep(50_000_000)
                self._block_floats(old_indices[-1]).fill_(2.0)
            kv.suspend(CachedCudaEvent(CudaStream(fwd.cuda_stream)))
        torch.cuda.synchronize()
        with TemporaryCudaStream([]) as s2:
            self.assertTrue(kv.resume(CudaStream(s2.handle)))
            torch.cuda.synchronize()
            indices = list(kv.get_base_page_indices(LifeCycleId(0)))
            tail = self._block_floats(indices[-1])
            self.assertTrue(bool((tail == 2.0).all().item()))
            kv.close()
        torch.cuda.synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL, wait=True)

    def test_suspend_drops_private_copies(self) -> None:
        """§4.5 x §4.4 interaction.

        Suspending a sequence whose prefix was onboarded from the radix tree
        swaps its private copies back to the canonical entries (no host
        duplication) and re-onboards them on resume.
        """
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        tokens = [500 + i for i in range(2 * self.TPB)]
        kv1 = manager.create_kv_cache(input_tokens=tokens, max_capacity=2 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(2 * self.TPB))
            for j, idx in enumerate(kv1.get_base_page_indices(LifeCycleId(0))):
                self._block_floats(idx).fill_(5.0 + j)
            torch.cuda.synchronize()
            kv1.commit(tokens)
            kv1.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        self.assertEqual(self._host_used(), 2)

        kv2 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv2.history_length, 2 * self.TPB)
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv2.resume(stream2))
            torch.cuda.synchronize()
            kv2.suspend()
        s2.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)
        self.assertEqual(self._host_used(), 2)  # privates dropped, no duplicates

        with TemporaryCudaStream([]) as s3:
            stream3 = CudaStream(s3.handle)
            self.assertTrue(kv2.resume(stream3))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv2.get_base_page_indices(LifeCycleId(0))):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 5.0 + j).all().item()))
            kv2.close()
        s3.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        self.assertEqual(self._host_used(), 2)


@requires_cuda
class TestArenaBatchedMapSweep(unittest.TestCase):
    """§4.2 batched per-iteration map sweep.

    Growth charges the budget at resize time but defers cuMemMap /
    cuMemSetAccess to flush_gpu_mappings.
    """

    TPB = 16

    def setUp(self) -> None:
        init_cuda_once()
        cfg = _make_manager_config(gpu_quota=32 * MiB, host_quota=32 * MiB)
        cfg.contiguous_arena = ContiguousArenaConfig(map_ahead_pages=0, batched_map_sweep=True)
        self.manager = KVCacheManager(cfg)

    def tearDown(self) -> None:
        self.manager.shutdown()
        del self.manager

    def _mapped_pages(self) -> int:
        return (
            self.manager._storage._gpu_arena_storage()
            .pool_group(PoolGroupIndex(0))
            ._arena.mapped_pages
        )

    def test_growth_defers_until_flush(self) -> None:
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        kv = manager.create_kv_cache(max_capacity=256 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(128 * self.TPB))  # 2 pages
            # budget charged (admission control unchanged), nothing mapped yet
            self.assertEqual(budget.used_pages, 2)
            self.assertEqual(self._mapped_pages(), 0)
            # growing again in the same iteration charges only the delta
            self.assertTrue(kv.resize(256 * self.TPB))  # 4 pages total
            self.assertEqual(budget.used_pages, 4)
            self.assertEqual(self._mapped_pages(), 0)
            self.assertEqual(manager.flush_gpu_mappings(), 4)
            self.assertEqual(self._mapped_pages(), 4)
            # pages are writable after the flush
            for idx in kv.get_base_page_indices(LifeCycleId(0))[:4]:
                addr = int(
                    manager._storage.slot_address(GPU_LEVEL, PoolGroupIndex(0), idx, PoolIndex(0))
                )
                TestSparseVirtMem._tensor_at(addr, 16).fill_(1.0)
            torch.cuda.synchronize()
            kv.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)

    def test_freed_before_flush_releases_charge(self) -> None:
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        kv = manager.create_kv_cache(max_capacity=128 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(128 * self.TPB))
            self.assertEqual(budget.used_pages, 2)
            kv.close()  # close() itself flushes (its write-out reads pages)
        s.take_finish_event().synchronize()
        manager.flush_gpu_mappings()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # ranges park for adoption on drain; spill to assert full release
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)

    def test_reuse_onboarding_stays_synchronous(self) -> None:
        manager = self.manager
        tokens = [970 + i for i in range(2 * self.TPB)]
        kv1 = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(2 * self.TPB))
            manager.flush_gpu_mappings()
            for j, idx in enumerate(kv1.get_base_page_indices(LifeCycleId(0))):
                addr = int(
                    manager._storage.slot_address(GPU_LEVEL, PoolGroupIndex(0), idx, PoolIndex(0))
                )
                TestSparseVirtMem._tensor_at(addr, (4 * 8192) // 4).fill_(3.0 + j)
            torch.cuda.synchronize()
            kv1.commit(tokens)
            kv1.close()
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
        # reuse hit: onboarding maps synchronously (its copies run immediately)
        kv2 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv2.history_length, 2 * self.TPB)
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv2.resume(stream2))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv2.get_base_page_indices(LifeCycleId(0))):
                addr = int(
                    manager._storage.slot_address(GPU_LEVEL, PoolGroupIndex(0), idx, PoolIndex(0))
                )
                data = TestSparseVirtMem._tensor_at(addr, (4 * 8192) // 4)
                self.assertTrue(bool((data == 3.0 + j).all().item()))
            kv2.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)


@requires_cuda
class TestArenaLazyRetention(unittest.TestCase):
    """§4.4 phase 2 (P2): lazy GPU retention.

    Freed ranges stay mapped on a retained LRU; reuse hits copy D2D from the
    resident "ghost" bytes; pressure spills LRU-first; the resume gate counts
    retained pages as available.
    """

    TPB = 16
    WRITE_THROUGH = WriteThroughPolicy.ON_FREE

    def setUp(self) -> None:
        init_cuda_once()
        cfg = _make_manager_config(gpu_quota=32 * MiB, host_quota=32 * MiB)
        cfg.contiguous_arena = ContiguousArenaConfig(
            map_ahead_pages=0, lazy_gpu_retention=True, write_through=self.WRITE_THROUGH
        )
        self.manager = KVCacheManager(cfg)

    def tearDown(self) -> None:
        self.manager.shutdown()
        del self.manager

    def _block_floats(self, page_index: int) -> "torch.Tensor":
        addr = int(
            self.manager._storage.slot_address(
                GPU_LEVEL, PoolGroupIndex(0), page_index, PoolIndex(0)
            )
        )
        return TestSparseVirtMem._tensor_at(addr, (4 * 8192) // 4)

    def _host_floats(self, slot_id: int) -> "torch.Tensor":
        import ctypes

        addr = int(
            self.manager._storage.slot_address(
                CacheLevel(1), PoolGroupIndex(0), slot_id, PoolIndex(0)
            )
        )
        buf = (ctypes.c_float * ((4 * 8192) // 4)).from_address(addr)
        return torch.frombuffer(buf, dtype=torch.float32)

    def _commit_and_close(self, tokens, value: float):
        manager = self.manager
        kv = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(2 * self.TPB))
            for j, idx in enumerate(kv.get_base_page_indices(LifeCycleId(0))):
                self._block_floats(idx).fill_(value + j)
            torch.cuda.synchronize()
            kv.commit(tokens)
            kv.close()
        s.take_finish_event().synchronize()

    def test_retention_keeps_pages_utilization_reports_free(self) -> None:
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        tokens = [900 + i for i in range(2 * self.TPB)]
        self._commit_and_close(tokens, 11.0)
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)  # parked, not unmapped
        self.assertGreater(budget.used_pages, 0)
        self.assertEqual(budget.retained_pages, budget.used_pages)
        # retained pages count as available for the resume gate
        self.assertEqual(manager._storage.get_utilization(GPU_LEVEL), [0.0])

    def test_reuse_onboards_d2d_from_ghost(self) -> None:
        manager = self.manager
        tokens = [910 + i for i in range(2 * self.TPB)]
        self._commit_and_close(tokens, 21.0)
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # corrupt the HOST copies: if onboarding reads from host, the new
        # sequence sees junk; correct data proves the D2D ghost path
        kv2 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv2.history_length, 2 * self.TPB)
        for ordinal in range(2):
            host_id = kv2._blocks[ordinal].pages[0][LifeCycleId(0)].page.slot_id
            self._host_floats(host_id).fill_(-999.0)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv2.resume(stream))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv2.get_base_page_indices(LifeCycleId(0))):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 21.0 + j).all().item()), f"block {j} not D2D")
            kv2.close()
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def test_pressure_spills_retained_and_falls_back_to_host(self) -> None:
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        tokens = [920 + i for i in range(2 * self.TPB)]
        self._commit_and_close(tokens, 31.0)
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        retained_before = budget.retained_pages
        self.assertGreater(retained_before, 0)
        # a large fresh sequence needs (nearly) the whole 16-page budget:
        # mapping must succeed by spilling the retained range
        kv2 = manager.create_kv_cache(max_capacity=1024 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv2.resume(stream))
            self.assertTrue(kv2.resize(1024 * self.TPB))
            self.assertEqual(budget.retained_pages, 0)  # spilled
            kv2.close()
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
        # the ghost is gone; reuse now falls back to the (intact) host copy
        kv3 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv3.history_length, 2 * self.TPB)
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv3.resume(stream2))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv3.get_base_page_indices(LifeCycleId(0))):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 31.0 + j).all().item()))
            kv3.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def test_private_copies_refresh_ghosts(self) -> None:
        """Ghost refresh by private copies.

        A hot prefix's D2D source must survive its original committer's range
        being spilled: every closing user re-registers its private copy as
        the (fresher) ghost.
        """
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        tokens = [940 + i for i in range(2 * self.TPB)]
        self._commit_and_close(tokens, 41.0)  # kv1: canonical + oldest ghost
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # kv2 reuses (D2D) and closes: its private copies refresh the ghosts
        kv2 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv2.resume(stream))
            kv2.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # spill ONLY the LRU (kv1's) range; kv2's stays retained
        self.assertGreater(manager._storage.spill_gpu_retained(1), 0)
        self.assertGreater(budget.retained_pages, 0)
        # corrupt host copies; correct data proves the refreshed D2D ghost
        kv3 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv3.history_length, 2 * self.TPB)
        for ordinal in range(2):
            host_id = kv3._blocks[ordinal].pages[0][LifeCycleId(0)].page.slot_id
            self._host_floats(host_id).fill_(-888.0)
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv3.resume(stream2))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv3.get_base_page_indices(LifeCycleId(0))):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 41.0 + j).all().item()), f"block {j} lost ghost")
            kv3.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def test_va_pressure_spills_retained(self) -> None:
        manager = self.manager
        # VA capacity is 8192 blocks; three 4096-block reservations only fit
        # if the retained first range is spilled for its VA
        blocks = 4096
        kv1 = manager.create_kv_cache(max_capacity=blocks * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(2 * self.TPB))
            kv1.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)  # retained (holds VA)
        kv2 = manager.create_kv_cache(max_capacity=blocks * self.TPB)
        kv3 = manager.create_kv_cache(max_capacity=blocks * self.TPB)
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv2.resume(stream2))
            self.assertTrue(kv3.resume(stream2))  # requires spilling kv1's VA
            kv2.close()
            kv3.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)


@requires_cuda
class TestArenaLazyRetentionOnCommit(TestArenaLazyRetention):
    """P2 x §4.3 ON_COMMIT: adopted host copies register ghosts too."""

    WRITE_THROUGH = WriteThroughPolicy.ON_COMMIT


@requires_cuda
class TestArenaWriteThroughOnCommit(TestArenaKVCacheManagerEndToEnd):
    """The §4.3 ON_COMMIT write-through policy.

    Every end-to-end scenario must behave identically (inherited tests);
    additionally, copies happen at commit time and the free path adopts them
    instead of copying.
    """

    def setUp(self) -> None:
        init_cuda_once()
        cfg = _make_manager_config(gpu_quota=32 * MiB, host_quota=32 * MiB)
        cfg.contiguous_arena = ContiguousArenaConfig(
            map_ahead_pages=0, write_through=WriteThroughPolicy.ON_COMMIT
        )
        self.manager = KVCacheManager(cfg)

    def test_copies_at_commit_and_adopts_on_close(self) -> None:
        manager = self.manager
        tokens = [700 + i for i in range(2 * self.TPB)]
        kv = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(2 * self.TPB))
            for j, idx in enumerate(kv.get_base_page_indices(LifeCycleId(0))):
                self._block_floats(idx).fill_(7.0 + j)
            torch.cuda.synchronize()
            self.assertEqual(self._host_used(), 0)
            kv.commit(tokens)
            # write-through happened at commit: host copies exist while the
            # blocks are still live on the GPU
            self.assertEqual(self._host_used(), 2)
            wt_ids = sorted(slot.slot_id for slot in kv._write_through_slots.values())
            self.assertEqual(len(wt_ids), 2)
            kv.close()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # the free path adopted the write-through copies: no new host slots
        self.assertEqual(self._host_used(), 2)
        # the canonical pages now live in exactly those pre-copied slots
        kv2 = manager.create_kv_cache(input_tokens=tokens, max_capacity=4 * self.TPB)
        self.assertEqual(kv2.history_length, 2 * self.TPB)
        host_ids = sorted(kv2._blocks[o].pages[0][LifeCycleId(0)].page.slot_id for o in range(2))
        self.assertEqual(host_ids, wt_ids)
        # and the data survived: onboard and verify
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv2.resume(stream2))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv2.get_base_page_indices(LifeCycleId(0))):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 7.0 + j).all().item()))
            kv2.close()
        s2.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        self.assertEqual(self._host_used(), 2)

    def test_suspend_adopts_write_through_copies(self) -> None:
        manager = self.manager
        kv = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(3 * self.TPB))
            for j, idx in enumerate(kv.get_base_page_indices(LifeCycleId(0))):
                self._block_floats(idx).fill_(9.0 + j)
            torch.cuda.synchronize()
            kv.commit([800 + i for i in range(2 * self.TPB)])
            self.assertEqual(self._host_used(), 2)  # written through at commit
            kv.suspend()
        s.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        # committed blocks adopted (still 2) + the uncommitted block moved (1)
        self.assertEqual(self._host_used(), 3)
        with TemporaryCudaStream([]) as s2:
            stream2 = CudaStream(s2.handle)
            self.assertTrue(kv.resume(stream2))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv.get_base_page_indices(LifeCycleId(0))):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 9.0 + j).all().item()))
            kv.close()
        s2.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)
        self.assertEqual(self._host_used(), 2)


@requires_cuda
class TestArenaPrefixAliasing(unittest.TestCase):
    """P3 prefix aliasing end-to-end at the manager seam.

    A closing canonical owner's fully-committed prefix pages are pinned in
    the canonical-span registry; a same-prefix admission adopts the parked
    range prefix-affinely (or alias-maps a fresh one) and reads the SAME
    physical bytes -- no H2D copy (proven by corrupting the host copies), no
    new budget charge for the span. Spilling the registry falls back to the
    host-copy path.
    """

    TPB = 16
    # One full 2 MiB chunk of 32 KiB coalesced block records.
    PREFIX_BLOCKS = 64

    def setUp(self) -> None:
        init_cuda_once()
        cfg = _make_manager_config(gpu_quota=32 * MiB, host_quota=64 * MiB)
        cfg.contiguous_arena = ContiguousArenaConfig(map_ahead_pages=0, prefix_aliasing=True)
        self.manager = KVCacheManager(cfg)

    def tearDown(self) -> None:
        self.manager.shutdown()
        del self.manager

    def _block_floats(self, page_index: int) -> "torch.Tensor":
        addr = int(
            self.manager._storage.slot_address(
                GPU_LEVEL, PoolGroupIndex(0), page_index, PoolIndex(0)
            )
        )
        return TestSparseVirtMem._tensor_at(addr, (4 * 8192) // 4)

    def _host_floats(self, slot_id: int) -> "torch.Tensor":
        import ctypes

        addr = int(
            self.manager._storage.slot_address(
                CacheLevel(1), PoolGroupIndex(0), slot_id, PoolIndex(0)
            )
        )
        buf = (ctypes.c_float * ((4 * 8192) // 4)).from_address(addr)
        return torch.frombuffer(buf, dtype=torch.float32)

    def _pool_group(self):
        return self.manager._storage._gpu_arena_storage().pool_group(PoolGroupIndex(0))

    def test_alias_roundtrip_zero_copy_and_affine_adoption(self) -> None:
        manager = self.manager
        budget = manager._storage.gpu_page_budget
        pg = self._pool_group()
        n_tok = self.PREFIX_BLOCKS * self.TPB
        tokens = [1000 + i for i in range(n_tok)]

        # Owner A: write patterns, commit the full-chunk prefix, close.
        kv_a = manager.create_kv_cache(max_capacity=(self.PREFIX_BLOCKS + 4) * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv_a.resume(stream))
            self.assertTrue(kv_a.resize(n_tok))
            a_indices = list(kv_a.get_base_page_indices(LifeCycleId(0)))[: self.PREFIX_BLOCKS]
            for j, idx in enumerate(a_indices):
                self._block_floats(idx).fill_(float(j))
            torch.cuda.synchronize()
            kv_a.commit(tokens)
            kv_a.close()
        s.take_finish_event().synchronize()
        self.assertEqual(pg.canonical_span_pages, 1)  # 64 x 32 KiB = 1 chunk
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)  # parked
        used_after_a = budget.used_pages

        # B: same-prefix admission. Corrupt every host copy first: correct
        # GPU bytes can then only have come through the alias mapping.
        kv_b = manager.create_kv_cache(
            input_tokens=tokens + [1, 2], max_capacity=(self.PREFIX_BLOCKS + 4) * self.TPB
        )
        self.assertEqual(kv_b.history_length, n_tok)
        for ordinal in range(self.PREFIX_BLOCKS):
            host_id = kv_b._blocks[ordinal].pages[0][LifeCycleId(0)].page.slot_id
            self._host_floats(host_id).fill_(-777.0)
        with TemporaryCudaStream([]) as s2:
            self.assertTrue(kv_b.resume(CudaStream(s2.handle)))
            self.assertEqual(pg.alias_hits, 1)
            torch.cuda.synchronize()
            b_indices = list(kv_b.get_base_page_indices(LifeCycleId(0)))[: self.PREFIX_BLOCKS]
            for j, idx in enumerate(b_indices):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == float(j)).all().item()), f"block {j} not aliased")
            kv_b.close()
        s2.take_finish_event().synchronize()
        self.assertEqual(manager.drain_gpu_reclaim(CachedCudaEvent.NULL), 1)

        # C: adoption is prefix-affine -- C inherits a parked same-signature
        # range with the alias mappings intact (no new maps for the span).
        kv_c = manager.create_kv_cache(
            input_tokens=tokens + [3, 4], max_capacity=(self.PREFIX_BLOCKS + 4) * self.TPB
        )
        with TemporaryCudaStream([]) as s3:
            self.assertTrue(kv_c.resume(CudaStream(s3.handle)))
            self.assertEqual(pg.alias_hits, 2)
            self.assertIsNotNone(kv_c._arena_ranges[PoolGroupIndex(0)]._alias_span_key)
            torch.cuda.synchronize()
            c_indices = list(kv_c.get_base_page_indices(LifeCycleId(0)))[: self.PREFIX_BLOCKS]
            for j, idx in enumerate(c_indices):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == float(j)).all().item()), f"block {j} lost")
            kv_c.close()
        s3.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
        self.assertLessEqual(budget.used_pages, used_after_a + 2)

        # Pressure spill drops parked ranges AND the registry pin; the next
        # same-prefix admission misses and falls back to the (corrupted)
        # host copies -- proving the fallback path engages.
        torch.cuda.synchronize()
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(budget.used_pages, 0)
        self.assertEqual(pg.canonical_span_pages, 0)
        kv_d = manager.create_kv_cache(
            input_tokens=tokens + [5, 6], max_capacity=(self.PREFIX_BLOCKS + 4) * self.TPB
        )
        with TemporaryCudaStream([]) as s4:
            self.assertTrue(kv_d.resume(CudaStream(s4.handle)))
            self.assertEqual(pg.alias_hits, 2)  # no new hit
            self.assertGreater(pg.alias_misses, 0)
            torch.cuda.synchronize()
            d_indices = list(kv_d.get_base_page_indices(LifeCycleId(0)))[: self.PREFIX_BLOCKS]
            data = self._block_floats(d_indices[0])
            self.assertTrue(bool((data == -777.0).all().item()), "fallback not from host")
            kv_d.close()
        s4.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def test_stale_span_falls_back_to_copy(self) -> None:
        """Spill between an admission's lookup hit and its first resume.

        The hit is held unpinned across that window; pressure can spill the
        registry in between, so the dead span must fall back to the
        host-copy path instead of alias-mapping freed handles (this exact
        shape crashed the tinyhost pressure run).
        """
        manager = self.manager
        pg = self._pool_group()
        n_tok = self.PREFIX_BLOCKS * self.TPB
        tokens = [5000 + i for i in range(n_tok)]
        kv_a = manager.create_kv_cache(max_capacity=n_tok + 4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            self.assertTrue(kv_a.resume(CudaStream(s.handle)))
            self.assertTrue(kv_a.resize(n_tok))
            for j, idx in enumerate(
                list(kv_a.get_base_page_indices(LifeCycleId(0)))[: self.PREFIX_BLOCKS]
            ):
                self._block_floats(idx).fill_(float(j))
            torch.cuda.synchronize()
            kv_a.commit(tokens)
            kv_a.close()
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

        # Admission: lookup HIT stored (unpinned) on the kv cache.
        kv_b = manager.create_kv_cache(input_tokens=tokens + [1], max_capacity=n_tok + 4 * self.TPB)
        self.assertEqual(pg.alias_hits, 1)
        # Pressure: everything spills, registry pins drop to zero refs.
        torch.cuda.synchronize()
        manager._storage.spill_gpu_retained(1 << 62)
        self.assertEqual(pg.canonical_span_pages, 0)
        with TemporaryCudaStream([]) as s2:
            self.assertTrue(kv_b.resume(CudaStream(s2.handle)))  # must not crash
            self.assertIsNone(kv_b._arena_ranges[PoolGroupIndex(0)]._alias_span_key)
            torch.cuda.synchronize()
            # intact host copies served the fallback -> bytes are correct
            for j, idx in enumerate(
                list(kv_b.get_base_page_indices(LifeCycleId(0)))[: self.PREFIX_BLOCKS]
            ):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == float(j)).all().item()), f"block {j} wrong")
            kv_b.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def test_alias_partial_span_from_longer_owner(self) -> None:
        """P3 partial-span aliasing: the shared-prompt benchmark shape.

        The canonical owner's committed chain is LONGER than the shared
        prefix (system prompt + its own continuation), so a later
        admission's match covers only part of the registered span. The
        match aliases exactly its own fully-covered chunks and writes
        strictly past them; the parked signature carries the extent so
        adoption stays exact, and the owner's tail chunks are never
        corrupted by a shorter adopter.
        """
        manager = self.manager
        pg = self._pool_group()
        shared_tok = self.PREFIX_BLOCKS * self.TPB  # 64 blocks = 1 chunk
        owner_tok = 2 * shared_tok  # owner commits 128 blocks = 2 chunks
        tokens = [3000 + i for i in range(owner_tok)]

        kv_a = manager.create_kv_cache(max_capacity=owner_tok + 4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            self.assertTrue(kv_a.resume(CudaStream(s.handle)))
            self.assertTrue(kv_a.resize(owner_tok))
            a_indices = list(kv_a.get_base_page_indices(LifeCycleId(0)))
            for j, idx in enumerate(a_indices[: 2 * self.PREFIX_BLOCKS]):
                self._block_floats(idx).fill_(float(j))
            torch.cuda.synchronize()
            kv_a.commit(tokens)
            kv_a.close()
        s.take_finish_event().synchronize()
        self.assertEqual(pg.canonical_span_pages, 2)  # the full 128-block span
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

        # B shares only the first 64 blocks; its match is HALF the span.
        kv_b = manager.create_kv_cache(
            input_tokens=tokens[:shared_tok] + [1, 2],
            max_capacity=owner_tok + 4 * self.TPB,
        )
        self.assertEqual(kv_b.history_length, shared_tok)
        for ordinal in range(self.PREFIX_BLOCKS):
            host_id = kv_b._blocks[ordinal].pages[0][LifeCycleId(0)].page.slot_id
            self._host_floats(host_id).fill_(-555.0)
        with TemporaryCudaStream([]) as s2:
            self.assertTrue(kv_b.resume(CudaStream(s2.handle)))
            self.assertEqual(pg.alias_hits, 1)
            rng = kv_b._arena_ranges[PoolGroupIndex(0)]
            self.assertEqual(rng._alias_span_key[1], self.PREFIX_BLOCKS)  # extent
            torch.cuda.synchronize()
            b_indices = list(kv_b.get_base_page_indices(LifeCycleId(0)))
            for j, idx in enumerate(b_indices[: self.PREFIX_BLOCKS]):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == float(j)).all().item()), f"block {j} not aliased")
            # B writes past the aliased extent: its block 64 lives in a
            # FRESH chunk of its own range, not the owner's second chunk.
            self.assertTrue(kv_b.resize(shared_tok + self.TPB))
            tail_idx = list(kv_b.get_base_page_indices(LifeCycleId(0)))[self.PREFIX_BLOCKS]
            self._block_floats(tail_idx).fill_(-1.0)
            torch.cuda.synchronize()
            kv_b.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

        # C matches the FULL owner chain: adoption is extent-exact (it takes
        # the owner's (key, 128) range, not B's (key, 64) one), and the
        # owner's tail chunk was never corrupted by B's shorter alias.
        kv_c = manager.create_kv_cache(
            input_tokens=tokens + [9], max_capacity=owner_tok + 4 * self.TPB
        )
        self.assertEqual(kv_c.history_length, owner_tok)
        with TemporaryCudaStream([]) as s3:
            self.assertTrue(kv_c.resume(CudaStream(s3.handle)))
            self.assertEqual(pg.alias_hits, 2)
            torch.cuda.synchronize()
            c_indices = list(kv_c.get_base_page_indices(LifeCycleId(0)))
            for j in (0, self.PREFIX_BLOCKS, 2 * self.PREFIX_BLOCKS - 1):
                data = self._block_floats(c_indices[j])
                self.assertTrue(bool((data == float(j)).all().item()), f"block {j} corrupted")
            kv_c.close()
        s3.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)


@requires_cuda
class TestArenaSuspendHostExhaustion(unittest.TestCase):
    """§4.5 hardening: host-tier exhaustion during suspension.

    The host tier fills with already-suspended sequences' HELD pages, which
    LRU eviction cannot drop at the last level (``is_evictable``); a further
    suspend() must then raise ``OutOfPagesError`` BEFORE mutating anything
    (the pre-flight reservation in ``suspend``), leaving the victim fully
    ACTIVE so the caller can degrade to drop-and-recompute. Stale DROPPABLE
    host copies, by contrast, must keep churning through LRU eviction —
    pressure means reuse loss, never a crash.
    """

    TPB = 16  # tokens per block

    def setUp(self) -> None:
        init_cuda_once()
        # Host tier: exactly 4 block slots (4 x 32 KiB coalesced records).
        cfg = _make_manager_config(gpu_quota=32 * MiB, host_quota=4 * 32 * 1024)
        cfg.contiguous_arena = ContiguousArenaConfig(map_ahead_pages=0)
        self.manager = KVCacheManager(cfg)

    def tearDown(self) -> None:
        self.manager.shutdown()
        del self.manager

    def _block_floats(self, page_index: int) -> "torch.Tensor":
        addr = int(
            self.manager._storage.slot_address(
                GPU_LEVEL, PoolGroupIndex(0), page_index, PoolIndex(0)
            )
        )
        return TestSparseVirtMem._tensor_at(addr, (4 * 8192) // 4)

    def _host_used(self) -> int:
        stats = self.manager._storage.get_statistics(CacheLevel(1))[0]
        return stats.total - stats.free

    def test_suspend_raises_atomically_when_host_pinned(self) -> None:
        manager = self.manager
        kv_a = manager.create_kv_cache(max_capacity=4 * self.TPB)
        kv_b = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv_a.resume(stream))
            self.assertTrue(kv_a.resize(3 * self.TPB))
            self.assertTrue(kv_b.resume(stream))
            self.assertTrue(kv_b.resize(3 * self.TPB))
            b_indices = list(kv_b.get_base_page_indices(LifeCycleId(0)))
            for j, idx in enumerate(b_indices):
                self._block_floats(idx).fill_(10.0 + j)
            torch.cuda.synchronize()
            kv_a.commit([100 + i for i in range(2 * self.TPB)])
            kv_b.commit([200 + i for i in range(2 * self.TPB)])
            # A's suspension takes 3 of the 4 host slots as HELD state.
            kv_a.suspend()
            self.assertEqual(self._host_used(), 3)
            # B needs 3 slots; 1 free, and A's HELD pages are not evictable
            # at the last level -> pre-flight raises with B untouched.
            with self.assertRaises(OutOfPagesError):
                kv_b.suspend()
            # Atomicity: B is still ACTIVE, byte-identical, and usable.
            indices = list(kv_b.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, b_indices)
            for j, idx in enumerate(indices):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 10.0 + j).all().item()))
            self.assertTrue(kv_b.resize(4 * self.TPB))  # growth still works
            # close() under the same exhaustion drops the write-out (reuse
            # loss only) instead of raising -- the free path never errors.
            kv_b.close()
            self.assertEqual(self._host_used(), 3)  # nothing of B landed
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
        # A was not perturbed by B's failed suspend: resumes byte-... (A had
        # no patterns written; assert the roundtrip machinery instead).
        with TemporaryCudaStream([]) as s2:
            self.assertTrue(kv_a.resume(CudaStream(s2.handle)))
            torch.cuda.synchronize()
            indices = list(kv_a.get_base_page_indices(LifeCycleId(0)))
            self.assertEqual(indices, list(range(indices[0], indices[0] + 3)))
            kv_a.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)

    def test_suspend_churns_through_droppable_host_copies(self) -> None:
        """Pressure evicts stale DROPPABLE host copies, not suspends.

        Stale copies churn through LRU eviction (reuse loss), so a suspend
        that fits after eviction succeeds.
        """
        manager = self.manager
        kv_a = manager.create_kv_cache(max_capacity=4 * self.TPB)
        kv_b = manager.create_kv_cache(max_capacity=4 * self.TPB)
        with TemporaryCudaStream([]) as s:
            stream = CudaStream(s.handle)
            self.assertTrue(kv_a.resume(stream))
            self.assertTrue(kv_a.resize(3 * self.TPB))
            self.assertTrue(kv_b.resume(stream))
            self.assertTrue(kv_b.resize(3 * self.TPB))
            b_indices = list(kv_b.get_base_page_indices(LifeCycleId(0)))
            for j, idx in enumerate(b_indices):
                self._block_floats(idx).fill_(20.0 + j)
            torch.cuda.synchronize()
            kv_a.commit([100 + i for i in range(2 * self.TPB)])
            kv_b.commit([300 + i for i in range(2 * self.TPB)])
            # A: close (not suspend) -> its 2 committed blocks become stale
            # DROPPABLE host copies (LRU-evictable), not HELD state.
            kv_a.close()
            self.assertEqual(self._host_used(), 2)
            # B's suspension needs 3 slots; 2 free + A's 2 droppable -> LRU
            # evicts A's oldest copy (block 0, the tree parent), whose GC
            # collapses the now-unreachable child copy too (reuse loss), and
            # the suspend SUCCEEDS: only B's 3 HELD pages remain.
            kv_b.suspend()
            self.assertEqual(self._host_used(), 3)
        s.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)
        with TemporaryCudaStream([]) as s2:
            self.assertTrue(kv_b.resume(CudaStream(s2.handle)))
            torch.cuda.synchronize()
            for j, idx in enumerate(kv_b.get_base_page_indices(LifeCycleId(0))):
                data = self._block_floats(idx)
                self.assertTrue(bool((data == 20.0 + j).all().item()))
            kv_b.close()
        s2.take_finish_event().synchronize()
        manager.drain_gpu_reclaim(CachedCudaEvent.NULL)


if __name__ == "__main__":
    unittest.main()
