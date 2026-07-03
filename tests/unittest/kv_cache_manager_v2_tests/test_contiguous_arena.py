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
    from kv_cache_manager_v2 import ContiguousArenaConfig, WriteThroughPolicy
    from kv_cache_manager_v2._cuda_virt_mem import PooledPhysMemAllocator, VirtMem
    from kv_cache_manager_v2._exceptions import LogicError, OutOfPagesError
    from kv_cache_manager_v2._sequence_arena import (
        INT31_MAX,
        BlockRangeAllocator,
        PageBudget,
        SequenceArena,
        check_index_width,
    )
    from kv_cache_manager_v2._storage._core import ArenaPoolGroup, GpuArenaCacheLevelStorage
    from kv_cache_manager_v2._utils import CachedCudaEvent, init_cuda_once
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import ContiguousArenaConfig, WriteThroughPolicy
    from tensorrt_llm.runtime.kv_cache_manager_v2._cuda_virt_mem import (
        PooledPhysMemAllocator,
        VirtMem,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import LogicError, OutOfPagesError
    from tensorrt_llm.runtime.kv_cache_manager_v2._sequence_arena import (
        INT31_MAX,
        BlockRangeAllocator,
        PageBudget,
        SequenceArena,
        check_index_width,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import (
        ArenaPoolGroup,
        GpuArenaCacheLevelStorage,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import CachedCudaEvent, init_cuda_once

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
            self.assertEqual(group.drain_reclaim(), 1)
            self.assertEqual(group.mapped_pages, 0)
            self.assertEqual(self.budget.used_pages, 0)
        finally:
            group.destroy()

    def test_reclaim_gated_on_outstanding_slots(self) -> None:
        group = self._group()
        try:
            rng = group.reserve_sequence(2)
            group.ensure_mapped(rng, 2)
            slot = group.take_slot(rng, 0)
            group.free_sequence(rng, CachedCudaEvent.NULL)
            self.assertEqual(group.drain_reclaim(), 0)  # slot still outstanding
            group.release(slot)
            self.assertEqual(group.drain_reclaim(), 1)
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
            group.drain_reclaim()
        finally:
            group.destroy()


@requires_cuda
class TestGpuArenaCacheLevelStorage(unittest.TestCase):
    PAGE = 2 * MiB

    def setUp(self) -> None:
        init_cuda_once()

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
            self.assertEqual(storage.drain_reclaim(), 2)
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


if __name__ == "__main__":
    unittest.main()
