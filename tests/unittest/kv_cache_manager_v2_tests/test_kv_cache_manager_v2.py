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

import array
import contextlib
import functools
import gc
import hashlib
import itertools
import math
import os
import random
import sys
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from importlib.util import find_spec
from random import randbytes
from statistics import median
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Sequence, cast

import pytest

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        BAD_PAGE_INDEX,
        DEFAULT_BEAM_INDEX,
        DEFAULT_COLD_TIER_OFFLOAD_MIN_PRIORITY,
        GPU_LEVEL,
        AttentionLayerConfig,
        Batch,
        BatchDesc,
        BufferConfig,
        BufferId,
        CacheLevel,
        CacheTier,
        CorruptedError,
        CudaStream,
        DataRole,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        IndexTableSpec,
        KVCacheDesc,
        KVCacheEventManager,
        KVCacheManager,
        KVCacheManagerConfig,
        KVCacheUpdatedData,
        LayerGroupId,
        LayerId,
        MemAddress,
        OutOfPagesError,
        PageIndexMode,
        PlannedDropHandle,
        PoolRebalanceConfig,
        ReuseScope,
        SlidingWindowSize,
        SparseCacheConfig,
        SsmLayerConfig,
        SwaScratchReuseConfig,
        TokenId,
        TokenIdExt,
        _introspection,
        _KVCache,
        gen_multimodal_cache_key_tokens,
        num_live_managers,
        poison_reason,
        sequence_to_blockchain_keys,
        take_poison,
    )

    # Must follow kv_cache_manager_v2: fast mode puts only tensorrt_llm/runtime on
    # PYTHONPATH, so `bindings` becomes importable only once _load_cpp_module() has
    # walked up to the tensorrt_llm root. isort would otherwise sort it above.
    from bindings.internal.batch_manager.kv_cache_manager_v2_utils import (  # isort: skip
        MemToMemTask,
        copy_device_to_device,
    )
    from bindings.executor import KvCacheRetentionConfig, KvCacheTransferMode  # isort: skip
else:
    from tensorrt_llm.bindings.executor import KvCacheRetentionConfig, KvCacheTransferMode
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        MemToMemTask,
        copy_device_to_device,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        BAD_PAGE_INDEX,
        DEFAULT_BEAM_INDEX,
        DEFAULT_COLD_TIER_OFFLOAD_MIN_PRIORITY,
        GPU_LEVEL,
        AttentionLayerConfig,
        Batch,
        BatchDesc,
        BufferConfig,
        BufferId,
        CacheLevel,
        CacheTier,
        CorruptedError,
        CudaStream,
        DataRole,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        IndexTableSpec,
        KVCacheDesc,
        KVCacheEventManager,
        KVCacheManager,
        KVCacheManagerConfig,
        KVCacheUpdatedData,
        LayerGroupId,
        LayerId,
        MemAddress,
        OutOfPagesError,
        PageIndexMode,
        PlannedDropHandle,
        PoolRebalanceConfig,
        ReuseScope,
        SlidingWindowSize,
        SparseCacheConfig,
        SsmLayerConfig,
        SwaScratchReuseConfig,
        TokenId,
        TokenIdExt,
        _introspection,
        _KVCache,
        gen_multimodal_cache_key_tokens,
        num_live_managers,
        poison_reason,
        sequence_to_blockchain_keys,
        take_poison,
    )

from copy import deepcopy

from parameterized import parameterized

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# cuda_test_utils supplies temporary_sys_path, so its own path entry is added and
# removed by hand here; every later sibling import goes through that helper.
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from cuda_test_utils import (  # noqa: E402
        CachedCudaStream,
        HalfOpenRange,
        TemporaryCudaStream,
        div_up,
        exact_div,
        init_cuda_once,
        intersect,
        remove_if,
        round_up,
        temporary_sys_path,
        typed_range,
    )
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

with temporary_sys_path(_TEST_DIR):
    from fake_engine import FakeEngine, Role, Step
    from kernels import HostGate, enable_kernel_delay


# A Batch that is only inspected, never published, still has to name a stream. The null
# stream is the honest answer for those: nothing is ever enqueued on it.
NULL_STREAM = cast(CudaStream, 0)


class _ScratchDesc(NamedTuple):
    """A life cycle's scratch range and the slot ids serving it."""

    range: HalfOpenRange
    slot_ids: list[int]


def _scratch_desc(kv_cache, lg_id):
    """The scratch range and the slots serving it, or None when none of the blocks do."""
    scratch_range = kv_cache.get_swa_scratch_range(lg_id)
    if not scratch_range:
        return None
    return _ScratchDesc(scratch_range, kv_cache.get_scratch_slots(lg_id))


def get_cached_cuda_event_type():
    try:
        from bindings.internal.batch_manager.kv_cache_manager_v2 import _introspection

        return _introspection.CachedCudaEvent
    except ImportError:
        from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2 import _introspection

        return _introspection.CachedCudaEvent


seed = int.from_bytes(os.urandom(8), "little")
print(f"seed: {seed}")
random.seed(seed)
DBG_PRINT = int(os.environ.get("DBG_PRINT", "0")) != 0
PRINT_TIME = int(os.environ.get("PRINT_TIME", "0")) != 0


@contextmanager
def ref_cycle_check_context():
    """Context manager for reference cycle check."""
    import gc

    gc.collect()
    gc.garbage.clear()
    gc.set_debug(gc.DEBUG_SAVEALL | gc.DEBUG_COLLECTABLE)

    def on_gc_event(phase, info):
        # phase is "start" or "stop"
        # info contains keys like: "generation", "collected", "uncollectable", "duration"
        if phase == "stop":
            collected = info.get("collected", 0)
            uncollectable = info.get("uncollectable", 0)
            if collected != 0 or uncollectable != 0:
                import pdb

                pdb.set_trace()
            assert collected == 0 and uncollectable == 0

    gc.callbacks.append(on_gc_event)
    try:
        yield
    finally:
        gc.collect()
        gc.callbacks.pop()
        gc.set_debug(0)


def assert_no_ref_cycle(func):
    """Decorator to wrap test methods with GC debugging context."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with ref_cycle_check_context():
            result = func(self, *args, **kwargs)
        return result

    return wrapper


class TestCacheLevelStorage(unittest.TestCase):
    def test_grains_to_slots_rejects_zero_divisors(self) -> None:
        invalid_inputs = [
            (1, [0], 16 << 20),
            (1, [16 << 20], 0),
        ]

        for grains, slot_sizes, granularity in invalid_inputs:
            with self.subTest(slot_sizes=slot_sizes, granularity=granularity):
                with self.assertRaisesRegex((ValueError, RuntimeError), "must be positive"):
                    _introspection.grains_to_slots(grains, slot_sizes, granularity)

    def test_grains_to_slots_refines_proportional_lower_bound(self) -> None:
        granularity = 16 << 20
        slot_size_list = [16_252_928, 4_063_232]
        min_slots = 157

        grains = _introspection.grains_for_slots(min_slots, slot_size_list, granularity)
        slots, used = _introspection.grains_to_slots(grains, slot_size_list, granularity)

        self.assertGreaterEqual(slots, min_slots)
        self.assertLessEqual(used, grains)

    # One mebibyte per slot against a one-mebibyte grain, so a slot is a grain and the
    # arithmetic a reader has to follow is the sharing rather than the rounding.
    GRAIN = 1 << 20
    TWO_MEDIA = [1 << 20, 1 << 20]
    ONE_MEDIUM = [1 << 20]
    HOST = [1]

    def _split(self, quota, host_quota, min_slots, ratio, groups=2, device_only=()):
        """Slot counts for `groups` groups of one device pool and, usually, one host pool.

        A group named in `device_only` holds a device pool alone, which is how a group that
        costs the host budget nothing is really shaped -- the quota pays for every pool the
        host budget does not name, so there is no way to leave one unpaid.
        """
        return _introspection.ratio_to_slot_count_list(
            quota,
            [self.ONE_MEDIUM if g in device_only else self.TWO_MEDIA for g in range(groups)],
            ratio,
            self.GRAIN,
            min_slots,
            second_quota=host_quota,
            second_granularity=self.GRAIN,
            second_pools=[[] if g in device_only else self.HOST for g in range(groups)],
        )

    def test_a_group_with_no_pool_on_the_second_budget_is_never_bound_by_it(self) -> None:
        """An empty selector is how "this group costs that medium nothing" is said."""
        slots = self._split(
            quota=200 << 20,
            host_quota=10 << 20,
            min_slots=[1, 1],
            ratio=[0.5, 0.5],
            device_only=(0,),
        )
        self.assertGreater(slots[0], 10, "group 0 pays no host budget, so it is not held to it")
        self.assertLessEqual(slots[1], 10)

    def test_a_second_budget_nobody_exceeds_changes_nothing(self) -> None:
        generous = self._split(
            quota=60 << 20, host_quota=1000 << 20, min_slots=[1, 1], ratio=[0.5, 0.5]
        )
        # Without a second budget the quota pays for both pools, so the comparison is
        # against a one-pool group of the same device cost per slot.
        unbounded = _introspection.ratio_to_slot_count_list(
            60 << 20, [self.ONE_MEDIUM] * 2, [0.5, 0.5], self.GRAIN, [1, 1]
        )
        self.assertEqual(generous, unbounded)

    def test_a_host_bound_group_releases_its_device_budget(self) -> None:
        """The reason this is one pass and not a quota followed by a clamp.

        Group 0 has no host pool and group 1 does. Starve the host budget and group 1 is
        held down -- the device grains it then cannot use must reach group 0 rather than
        going unspent, which is precisely what clamping afterwards failed to do.
        """
        held = self._split(
            quota=200 << 20,
            host_quota=4 << 20,
            min_slots=[1, 1],
            ratio=[0.5, 0.5],
            device_only=(0,),
        )
        free = self._split(
            quota=200 << 20,
            host_quota=1000 << 20,
            min_slots=[1, 1],
            ratio=[0.5, 0.5],
            device_only=(0,),
        )
        self.assertLessEqual(held[1], 4, "the host budget bound group 1")
        self.assertGreater(
            held[0], free[0], "and group 0 grew into the device budget group 1 gave up"
        )

    def test_both_budgets_are_kept_and_the_ratio_gives_way(self) -> None:
        """Both bounds hold exactly; only the ratio is negotiable."""
        for ratio in ([0.5, 0.5], [0.9, 0.1], [0.01, 0.99]):
            with self.subTest(ratio=ratio):
                slots = self._split(
                    quota=120 << 20, host_quota=60 << 20, min_slots=[5, 5], ratio=ratio
                )
                self.assertLessEqual(sum(slots), 60, "the host budget was never exceeded")
                self.assertLessEqual(sum(slots), 120, "nor the device quota")
                self.assertGreaterEqual(slots[0], 5)
                self.assertGreaterEqual(slots[1], 5)

    def test_a_second_budget_too_small_for_the_constraints_is_rejected(self) -> None:
        """Both budgets are hard, so their only true conflict is a configuration error.

        It names neither the ratio nor the device quota, which is what makes it a
        construction-time property: a rebalance changes those and cannot start failing.
        """
        with self.assertRaisesRegex((ValueError, RuntimeError), "cannot hold"):
            self._split(quota=200 << 20, host_quota=40 << 20, min_slots=[50, 50], ratio=[0.5, 0.5])

    def test_an_empty_pool_selector_is_refused_rather_than_hung(self) -> None:
        """grains_to_slots searches for a slot count; with no pool there is none to find.

        It used to run past SlotCount's range looking for one, which is a hang rather than
        an answer -- the kind of failure that is only ever seen as a stuck process.
        """
        with self.assertRaisesRegex((ValueError, RuntimeError), "at least one pool"):
            _introspection.grains_to_slots(16, [1 << 20], 1 << 20, pools=[])

    def test_slot_counts_are_maximal_for_the_grains_they_cost(self) -> None:
        """Grains -> slots -> grains has to be a fixed point, or sizing disagrees with itself.

        grains_to_slots can return fewer slots than the grains it consumed would buy, when
        those grains divide badly across a group's pools. A count left that way is short of
        what it already paid for, and is not the count a re-derivation from the same bytes
        returns -- which is what the bump at the end of the allocator exists to prevent.
        A second budget must not cost the primary one that property.
        """
        # Sizes that do not divide the grain evenly are where the under-count shows up.
        uneven = [[3 << 20, 5 << 20], [7 << 20, 1 << 20]]
        # The quota pays for whatever `second` does not name, so each case says only which
        # pools the second budget takes.
        cases = {
            "one budget": dict(second=None, second_quota=None),
            "two budgets, ample second": dict(second=[[1], [1]], second_quota=256 << 20),
            "one group outside the second budget": dict(second=[[1], []], second_quota=256 << 20),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                slots = _introspection.ratio_to_slot_count_list(
                    512 << 20,
                    uneven,
                    [0.5, 0.5],
                    self.GRAIN,
                    [1, 1],
                    second_quota=case["second_quota"],
                    second_granularity=self.GRAIN if case["second"] else None,
                    second_pools=case["second"],
                )
                for group, count in enumerate(slots):
                    taken = set(case["second"][group]) if case["second"] else set()
                    primary = [p for p in range(len(uneven[group])) if p not in taken]
                    cost = _introspection.grains_for_slots(
                        count, uneven[group], self.GRAIN, pools=primary
                    )
                    recovered, _ = _introspection.grains_to_slots(
                        cost, uneven[group], self.GRAIN, pools=primary
                    )
                    self.assertEqual(
                        recovered,
                        count,
                        f"{name}: group {group} holds {count} slots but its own grains buy "
                        f"{recovered}, so re-deriving from the same bytes would disagree",
                    )
                if case["second"]:
                    spent = sum(
                        _introspection.grains_for_slots(
                            slots[g], uneven[g], self.GRAIN, pools=case["second"][g]
                        )
                        for g in range(len(slots))
                    )
                    self.assertLessEqual(
                        spent * self.GRAIN,
                        case["second_quota"],
                        f"{name}: the bump spent past the second budget",
                    )

    def test_the_division_settles_promptly_on_mismatched_grains(self) -> None:
        """A dead loop is not the only way the greedy finish can go wrong.

        A dense pool far smaller than the device grain beside an unbudgeted host pool is the
        shape where one device-grain step spans thousands of host-grain steps, and a step that
        costs the quota nothing cannot be refused on budget grounds. Such a loop still
        terminates, so only a clock catches it -- an assertion on the slot counts never will.
        """
        sizes = [[256, 1024], [4096]]
        start = time.monotonic()
        slots = _introspection.ratio_to_slot_count_list(
            1 << 30,
            sizes,
            [0.5, 0.5],
            32 << 20,
            [1, 1],
            second_quota=None,
            second_granularity=4096,
            second_pools=[[1], []],
        )
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed,
            2.0,
            f"the division took {elapsed:.1f}s, so it is stepping rather than jumping",
        )
        self.assertGreater(slots[0], 0)
        self.assertGreater(slots[1], 0)

    def _all_bytes(self, slots, sizes, second_pools=None, second_granularity=None):
        """What a slot count costs, each pool in the grain of the medium backing it.

        The measure CacheLevelStorage::allBytes() reports, rebuilt from the two selectors
        the allocator is given.
        """
        taken = set(second_pools or ())
        primary = [p for p in range(len(sizes)) if p not in taken]
        total = (
            _introspection.grains_for_slots(slots, sizes, self.GRAIN, pools=primary) * self.GRAIN
        )
        if taken:
            total += (
                _introspection.grains_for_slots(
                    slots, sizes, second_granularity, pools=sorted(taken)
                )
                * second_granularity
            )
        return total

    def test_feeding_back_the_layouts_own_ratio_changes_nothing(self) -> None:
        """The allocator must be a fixed point of its own output, or rebalancing never stops.

        _needAdjustment compares the ratio a rebalance would target against the ratio the
        current layout reports, and fires when they differ. Both are all-bytes measures, so
        if the slot counts this returns imply a ratio that it would answer differently, the
        manager re-adjusts to the same layout forever on an unchanged profile.
        """
        cases = {
            "single budget, uneven sizes": dict(
                sizes=[[3 << 20, 5 << 20], [7 << 20, 1 << 20], [2 << 20]],
                ratio=[0.2, 0.5, 0.3],
                min_slots=[1, 1, 1],
                quota=512 << 20,
                second_quota=None,
                second_pools=None,
            ),
            # The case the device-only ratio got wrong: a host-backed pool is most of the
            # group's bytes and none of its quota, so the two measures diverge sharply.
            "host-backed pool dominates the bytes": dict(
                sizes=[[1 << 20, 64 << 20], [1 << 20]],
                ratio=[0.5, 0.5],
                min_slots=[1, 1],
                quota=200 << 20,
                second_quota=4096 << 20,
                second_pools=[[1], []],
            ),
            "second budget binds": dict(
                sizes=[[1 << 20, 8 << 20], [1 << 20, 1 << 20]],
                ratio=[0.7, 0.3],
                min_slots=[2, 2],
                quota=400 << 20,
                second_quota=64 << 20,
                second_pools=[[1], [1]],
            ),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                second_granularity = self.GRAIN if case["second_pools"] else None

                def solve(ratio, case=case, second_granularity=second_granularity):
                    return _introspection.ratio_to_slot_count_list(
                        case["quota"],
                        case["sizes"],
                        ratio,
                        self.GRAIN,
                        case["min_slots"],
                        second_quota=case["second_quota"],
                        second_granularity=second_granularity,
                        second_pools=case["second_pools"],
                    )

                first = solve(case["ratio"])
                achieved = [
                    self._all_bytes(
                        first[g],
                        case["sizes"][g],
                        case["second_pools"][g] if case["second_pools"] else None,
                        second_granularity,
                    )
                    for g in range(len(first))
                ]
                total = sum(achieved)
                again = solve([b / total for b in achieved])
                self.assertEqual(
                    again,
                    first,
                    f"{name}: {first} reports ratio {[b / total for b in achieved]}, which "
                    f"the allocator answers with {again} -- a rebalance that never settles",
                )

    def test_ratio_to_slot_count_list_preserves_min_slots(self) -> None:
        granularity = 16 << 20
        slot_size_lists = [
            [16_252_928, 4_063_232],
            [491_520, 126_720, 15_872],
            [31_457_280, 7_864_320],
        ]
        min_slots = [157, 3907, 157]
        total_min_grains = sum(
            _introspection.grains_for_slots(slots, sizes, granularity)
            for slots, sizes in zip(min_slots, slot_size_lists)
        )

        slot_counts = _introspection.ratio_to_slot_count_list(
            total_min_grains * granularity,
            slot_size_lists,
            [0.2, 0.5, 0.3],
            granularity,
            min_slots,
        )

        for slot_count, min_slot in zip(slot_counts, min_slots):
            self.assertGreaterEqual(slot_count, min_slot)


def create_config(
    tokens_per_block: int,
    gpu_quota: int,
    host_quota: int,
    disk_quota: int,
    num_layers: int,
    window_size: SlidingWindowSize,
    sink_tokens: int,
    kv_buf_size: int = 8192,
    block_quant_buf_size: int | None = None,
) -> KVCacheManagerConfig:
    layer_buffers = [
        BufferConfig(role=Role.KEY, size=kv_buf_size),
        BufferConfig(role=Role.VALUE, size=kv_buf_size),
    ]
    if block_quant_buf_size is not None:
        layer_buffers.extend(
            [
                BufferConfig(role=Role.KEY_BLOCK_QUANT, size=block_quant_buf_size),
                BufferConfig(role=Role.VALUE_BLOCK_QUANT, size=block_quant_buf_size),
            ]
        )
    disk_path_candidates = ["/workspace/", "/tmp/nvidia-mps/", "/tmp"]
    disk_path = next(p for p in disk_path_candidates if os.path.exists(p))
    assert gpu_quota > 0
    cache_tiers = [
        GpuCacheTierConfig(quota=gpu_quota),
        HostCacheTierConfig(quota=host_quota),
        DiskCacheTierConfig(quota=disk_quota, path=disk_path),
    ]
    cache_tiers = [t for t in cache_tiers if t.quota > 0]
    return KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        cache_tiers=[t for t in cache_tiers if t.quota > 0],
        layers=[
            AttentionLayerConfig(
                layer_id=layer_id,
                buffers=deepcopy(layer_buffers),
                sliding_window_size=window_size if layer_id % 2 == 0 else None,
                num_sink_tokens=sink_tokens if layer_id % 2 == 0 else None,
            )
            for layer_id in typed_range(LayerId(num_layers))
        ],
    )


class TestSparseBufferConfig(unittest.TestCase):
    def test_flag_defaults_and_copy(self):
        self.assertFalse(BufferConfig("key", 4096).is_sparse)
        buffer = BufferConfig("key", 4096, 2, is_sparse=True)
        copied = deepcopy(buffer)
        self.assertTrue(copied.is_sparse)
        self.assertEqual(copied.tokens_per_block_override, 2)
        buffer.is_sparse = False
        self.assertTrue(copied.is_sparse)

    def test_the_quantised_shape_gives_each_pool_its_own_arena_region(self):
        """Data and block scales differ in size, so they land in pools of their own.

        The arena mirrors the hot pool group, so an index the fetch emits is relative to the
        region of the buffer it was emitted for -- which is why the base address is per
        buffer rather than per life cycle.
        """
        init_cuda_once()
        data = BufferConfig(
            "key",
            4096,
            is_sparse=True,
        )
        scales = BufferConfig(
            "key_scale",
            128,
            is_sparse=True,
        )
        config = KVCacheManagerConfig(
            tokens_per_block=128,
            cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=1024, history_length=0)] * 2)],
            # A pool group sizes its slots from its device-backed pools.
            layers=[
                AttentionLayerConfig(
                    0,
                    [data, scales, BufferConfig("index", 256)],
                    sparse_cache=SparseCacheConfig(
                        num_selected_blocks=2, max_cached_units=4, unit_tokens=128
                    ),
                )
            ],
        )
        manager = KVCacheManager(config)
        try:
            with TemporaryCudaStream([]) as stream:
                batch = Batch(
                    manager,
                    cast(CudaStream, stream.handle),
                    max_batch_size=1,
                    max_blocks_per_seq=1,
                    index_tables=manager.sparse_index_table_specs(),
                )
                bases = {
                    role: int(batch.get_sparse_cache_base_address(LayerId(0), role))
                    for role in ("key", "key_scale")
                }
            self.assertEqual(len(set(bases.values())), 2, bases)
        finally:
            manager.shutdown()

    def test_sparse_sizing_defaults_and_roundtrip(self):
        plain = BufferConfig("key", 4096)
        self.assertFalse(plain.is_sparse)
        self.assertEqual(plain.sparse_unit_chunks, 1)

        buffer = BufferConfig("key", 4096, is_sparse=True, sparse_unit_chunks=4)
        copied = deepcopy(buffer)
        buffer.sparse_unit_chunks = 1
        self.assertEqual(copied.sparse_unit_chunks, 4)
        self.assertTrue(copied.is_sparse)

        cache = SparseCacheConfig(num_selected_blocks=16, max_cached_units=64, unit_tokens=4)
        copied_cache = deepcopy(cache)
        cache.max_cached_units = 0
        self.assertEqual(copied_cache.max_cached_units, 64)
        self.assertEqual(copied_cache.num_selected_blocks, 16)
        self.assertEqual(copied_cache.unit_tokens, 4)

    def test_rejects_unit_chunks_on_a_dense_buffer(self):
        layer = AttentionLayerConfig(0, [BufferConfig("key", 4096, sparse_unit_chunks=2)])
        with self.assertRaisesRegex(RuntimeError, "sparseUnitChunks set on a non-sparse buffer"):
            self._make_config(layer)

    def test_rejects_a_cache_on_a_layer_with_no_sparse_buffer(self):
        """sparse_cache configures a fetch path, so there must be something to fetch for."""
        layer = AttentionLayerConfig(
            0,
            [BufferConfig("key", 4096)],
            sparse_cache=SparseCacheConfig(
                num_selected_blocks=2, max_cached_units=4, unit_tokens=128
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "no sparse buffer to serve"):
            self._make_config(layer)

    def test_rejects_cache_narrower_than_selection(self):
        layer = AttentionLayerConfig(
            0,
            [
                BufferConfig("key", 4096, is_sparse=True),
                # A pool group sizes its slots from its device-backed pools, so a sparse
                # layer carries one -- which is also the shape a model uses.
                BufferConfig("index", 256),
            ],
            sparse_cache=SparseCacheConfig(
                num_selected_blocks=16, max_cached_units=8, unit_tokens=128
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "maxCachedUnits"):
            self._make_config(layer)

    def test_rejects_a_page_its_units_do_not_divide(self):
        layer = AttentionLayerConfig(
            0,
            [
                BufferConfig(
                    "key",
                    4096,
                    is_sparse=True,
                    # 3 chunks do not divide 4096, so a unit has no whole-byte size.
                    sparse_unit_chunks=3,
                ),
                # A pool group sizes its slots from its device-backed pools, so a sparse
                # layer carries one -- which is also the shape a model uses.
                BufferConfig("index", 256),
            ],
            sparse_cache=SparseCacheConfig(
                num_selected_blocks=16, max_cached_units=64, unit_tokens=4
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "divide evenly"):
            self._make_config(layer)

    def test_rejects_both_unit_knobs_on_one_buffer(self):
        layer = AttentionLayerConfig(
            0,
            [
                BufferConfig(
                    "key",
                    4096,
                    tokens_per_block_override=4,
                    is_sparse=True,
                ),
                # A pool group sizes its slots from its device-backed pools, so a sparse
                # layer carries one -- which is also the shape a model uses.
                BufferConfig("index", 256),
            ],
            sparse_cache=SparseCacheConfig(
                num_selected_blocks=16, max_cached_units=64, unit_tokens=4
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "tokensPerBlockOverride is not supported"):
            self._make_config(layer)

    @staticmethod
    def _make_config(layer: "AttentionLayerConfig") -> "KVCacheManagerConfig":
        return KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[GpuCacheTierConfig(4 << 20), HostCacheTierConfig(4 << 20)],
            layers=[layer],
        )

    def test_rejects_a_layer_whose_buffers_are_all_sparse(self):
        """A slot is one row across a group's pools, priced in the device bytes it costs.

        Sparse pools are host memory and cost none, so a group holding only those has no
        slot count to derive. Raised when storage is built rather than by config
        validation: a pool group forms across layers, so no single layer can be judged.
        """
        layer = AttentionLayerConfig(
            0,
            [BufferConfig("key", 4096, is_sparse=True)],
            sparse_cache=SparseCacheConfig(
                num_selected_blocks=2, max_cached_units=4, unit_tokens=4
            ),
        )
        with self.assertRaisesRegex((ValueError, RuntimeError), "wholly sparse"):
            KVCacheManager(self._make_config(layer))

    def test_a_layer_may_mix_sparse_and_dense_buffers(self):
        """Sparsity picks a buffer's pool and nothing above it.

        An indexer and the data it selects over are one attention computation, so forcing
        them into separate layers to satisfy storage would be the tail wagging the dog.
        """
        layer = AttentionLayerConfig(
            0, [BufferConfig("key", 4096, is_sparse=True), BufferConfig("value", 4096)]
        )
        config = KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[GpuCacheTierConfig(4 << 20), HostCacheTierConfig(4 << 20)],
            # The cache arena is reserved up front, so a sparse deployment must say
            # how many requests run together.
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 4)],
            layers=[layer],
        )
        self.assertEqual([b.is_sparse for b in config.layers[0].buffers], [True, False])
        # Built, not just configured: the pool grouping that has to tolerate the mix runs
        # during construction, so a config-only check would pass even if storage refused it.
        manager = KVCacheManager(config)
        try:
            self.assertTrue(manager.is_sparse(0, "key"))
            self.assertFalse(manager.is_sparse(0, "value"))
        finally:
            manager.shutdown()


class TestSparseTierFlexibility(unittest.TestCase):
    """What a sparse deployment requires of its cold tiers: nothing.

    History lives on a host-backed pool group at the *hot* level, so level 1 is an ordinary
    codec target. It may be host, disk, or absent. An earlier revision required level 1 to
    be HOST_MEM, because that was where sparse history resided.
    """

    def setUp(self) -> None:
        init_cuda_once()

    @staticmethod
    def _manager(tiers) -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=128,
                cache_tiers=tiers,
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=1024, history_length=0)] * 2)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                "key",
                                4096,
                                is_sparse=True,
                            ),
                            # A pool group is sized from its device-backed pools, so a sparse layer
                            # carries one. This is also the shape a model uses: an indexer and the
                            # data it selects over are one attention computation.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=2, max_cached_units=4, unit_tokens=128
                        ),
                    )
                ],
            )
        )

    def test_any_cold_tier_arrangement_is_accepted(self):
        for name, tiers in (
            ("gpu only", [GpuCacheTierConfig(64 << 20)]),
            ("gpu + disk", [GpuCacheTierConfig(64 << 20), DiskCacheTierConfig(64 << 20, "/tmp")]),
            ("gpu + host", [GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)]),
        ):
            with self.subTest(tiers=name):
                manager = self._manager(tiers)
                try:
                    self.assertGreater(manager.get_sparse_cache_max_units(0), 0)
                finally:
                    manager.shutdown()

    @staticmethod
    def _manager_with_host_quota(tiers, host_quota) -> KVCacheManager:
        config = KVCacheManagerConfig(
            tokens_per_block=128,
            cache_tiers=tiers,
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=1024, history_length=0)] * 2)],
            layers=[
                AttentionLayerConfig(
                    0,
                    [
                        BufferConfig("key", 4096, is_sparse=True),
                        BufferConfig("index", 256),
                    ],
                    sparse_cache=SparseCacheConfig(
                        num_selected_blocks=2, max_cached_units=4, unit_tokens=128
                    ),
                )
            ],
            host_sparse_pool_quota=host_quota,
        )
        return KVCacheManager(config)

    @staticmethod
    def _sparse_pinned_bytes(manager: KVCacheManager) -> int:
        """Bytes the hot level holds in the medium backing its sparse buffers."""
        return manager.get_num_bytes(0, manager.sparse_backing_medium())

    def test_the_host_quota_bounds_what_the_sparse_pools_pin(self):
        """The host-backed pools have no budget of their own; this is the one that bounds them.

        Their row count follows from the device-backed pools the GPU quota sizes, so without
        this a deployment can only reach them by shrinking the device cache it does want.
        """
        tiers = [GpuCacheTierConfig(64 << 20)]
        unbounded = self._manager_with_host_quota(tiers, None)
        try:
            pinned = self._sparse_pinned_bytes(unbounded)
        finally:
            unbounded.shutdown()
        self.assertGreater(pinned, 0)

        capped = self._manager_with_host_quota(tiers, pinned // 2)
        try:
            self.assertLessEqual(self._sparse_pinned_bytes(capped), pinned // 2)
            self.assertLess(self._sparse_pinned_bytes(capped), pinned)
        finally:
            capped.shutdown()

    def test_a_host_quota_above_what_is_pinned_changes_nothing(self):
        tiers = [GpuCacheTierConfig(64 << 20)]
        unbounded = self._manager_with_host_quota(tiers, None)
        try:
            pinned = self._sparse_pinned_bytes(unbounded)
        finally:
            unbounded.shutdown()

        generous = self._manager_with_host_quota(tiers, pinned * 4)
        try:
            self.assertEqual(self._sparse_pinned_bytes(generous), pinned)
        finally:
            generous.shutdown()

    def test_a_zero_host_quota_is_refused(self):
        """Zero asks for host-backed pools with no room for a slot, which is a mistake.

        Whether a pool is host-backed is declared per buffer, so this is not the way to
        turn them off.
        """
        with self.assertRaisesRegex(Exception, "must be positive"):
            self._manager_with_host_quota([GpuCacheTierConfig(64 << 20)], 0)

    def test_a_sparse_pool_names_the_host_memory_it_actually_is(self):
        """A GPU-tier group is not uniform, so the medium has to be read per pool.

        The sparse pool's pages are host memory while the dense pool beside it is device
        memory, and the group reports one tier for both. A consumer that has to name the
        medium of an address -- registering it for a disaggregated transfer, say -- gets
        it wrong for exactly one pool if it asks the group.
        """
        manager = self._manager([GpuCacheTierConfig(64 << 20)])
        try:
            media = [
                pool.storage_medium for group in manager.pool_group_descs for pool in group.pools
            ]
            self.assertEqual(
                media.count(CacheTier.HOST_MEM), 1, "the one sparse buffer is the host-backed pool"
            )
            self.assertEqual(
                media.count(CacheTier.GPU_MEM),
                len(media) - 1,
                "every other pool of a GPU tier is device memory",
            )
        finally:
            manager.shutdown()


class TestSparseCacheReservation(unittest.TestCase):
    """GPU pages a request reserves to back its sparse history cache."""

    def setUp(self) -> None:
        init_cuda_once()

    @staticmethod
    def _manager(
        max_cached_units: int, *, num_selected_blocks: int = 2, unit_tokens: int = 128
    ) -> KVCacheManager:
        # sparse_cache is absent when the fetch path is unconfigured: the buffer still gets
        # sparse storage and locking, it simply has no GPU cache to read through.
        # The cache must exceed the selection, since the surplus is the in-flight headroom.
        sparse_cache = (
            SparseCacheConfig(
                num_selected_blocks=num_selected_blocks,
                max_cached_units=max_cached_units,
                unit_tokens=unit_tokens,
            )
            if max_cached_units > 0
            else None
        )
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=128,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                # The cache arena is reserved up front, so a sparse deployment must say
                # how many requests run together.
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 4)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                "key",
                                4096,
                                is_sparse=True,
                                # One contiguous run per unit; a cached buffer must say how
                                # to find a unit inside itself.
                            ),
                            # A pool group is sized from its device-backed pools, so a sparse layer
                            # carries one. This is also the shape a model uses: an indexer and the
                            # data it selects over are one attention computation.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=sparse_cache,
                    )
                ],
            )
        )

    @staticmethod
    def _free(manager: KVCacheManager) -> int:
        return sum(stats.free for stats in manager.get_storage_statistics())

    def test_batch_rejects_resolve_shared_memory_overflow_before_fetch(self) -> None:
        for max_cached_units, selection_width in ((1 << 20, 2), (1 << 16, (1 << 16) - 1)):
            with self.subTest(max_cached_units=max_cached_units, selection_width=selection_width):
                # Token-sized units keep the arena small while either residency metadata
                # or the selection's probe table exceeds a device's per-block limit.
                manager = self._manager(
                    max_cached_units,
                    num_selected_blocks=selection_width,
                    unit_tokens=1,
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, "above this device's limit"):
                        Batch(
                            manager,
                            NULL_STREAM,
                            max_batch_size=1,
                            max_blocks_per_seq=4,
                            index_tables=manager.sparse_index_table_specs(),
                        )
                finally:
                    manager.shutdown()

    def test_batch_accepts_resolve_shared_memory_that_fits(self) -> None:
        # Three token-sized units round up to a full 128-unit cell.
        manager = self._manager(max_cached_units=3, unit_tokens=1)
        batch = None
        try:
            batch = Batch(
                manager,
                NULL_STREAM,
                max_batch_size=1,
                max_blocks_per_seq=4,
                index_tables=manager.sparse_index_table_specs(),
            )
            self.assertEqual(batch.max_batch_size, 1)
        finally:
            del batch
            manager.shutdown()

    def _free_delta_across_resume(self, max_cached_units: int) -> int:
        """GPU slots a single resume consumes, for the given cache width."""
        manager = self._manager(max_cached_units)
        try:
            before = self._free(manager)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                self.assertTrue(cache.resume(cast(CudaStream, stream.handle)))
                used = before - self._free(manager)
                cache.close()
            self.assertEqual(self._free(manager), before)
            return used
        finally:
            manager.shutdown()

    def test_the_input_bound_is_one_token_whatever_the_cache_width(self):
        """A sparse request contributes one query token per step, so the gap is one token.

        The bound follows from the supported shape, not from how much spare room the
        reservation has: widening the cache buys reuse, not a longer input.
        """
        for width in (4, 6):
            manager = self._manager(max_cached_units=width)
            try:
                cache = manager.create_kv_cache()
                try:
                    self.assertEqual(cache.max_input_length, 1)
                finally:
                    cache.close()
            finally:
                manager.shutdown()

        dense = self._manager(max_cached_units=0)
        try:
            cache = dense.create_kv_cache()
            try:
                # None: without a cache to hold in-flight blocks the bound describes nothing.
                self.assertIsNone(cache.max_input_length)
            finally:
                cache.close()
        finally:
            dense.shutdown()

    def test_shape_constants_describe_one_buffer(self):
        manager = self._manager(max_cached_units=4)
        try:
            self.assertEqual(manager.get_sparse_cache_max_units(0), 4)
            self.assertEqual(manager.get_sparse_cache_units_per_page(0), 1)
            self.assertEqual(manager.get_sparse_cache_buffer_count(0), 1)
        finally:
            manager.shutdown()

    def test_the_cache_does_not_draw_on_the_storage_pool(self):
        """Cells come from the life cycle's own arena, so a resume costs the pool nothing.

        This is what lets the hot pool group become host memory without the cache following
        it there -- the two are no longer the same allocation.
        """
        baseline = self._free_delta_across_resume(0)
        for width in (4, 6):
            self.assertEqual(
                self._free_delta_across_resume(width),
                baseline,
                "a wider cache must not consume more storage slots",
            )

    def test_unconfigured_fetch_path_reserves_nothing(self):
        manager = self._manager(max_cached_units=0)
        try:
            self.assertEqual(manager.get_sparse_cache_max_units(0), 0)
        finally:
            manager.shutdown()

    def test_resize_rejects_more_than_one_token_in_flight(self):
        """The guard that keeps prefill and speculative decoding out.

        Either would give a request several query tokens in one step, which the fetch
        cannot serve: it resolves one selection per request, and the units a whole step's
        worth of query tokens wanted could not all be made resident at once.
        """
        manager = self._manager(max_cached_units=4)
        try:
            cache = manager.create_kv_cache()
            try:
                with TemporaryCudaStream([]) as stream:
                    self.assertTrue(cache.resume(cast(CudaStream, stream.handle)))
                    # One token ahead of the watermark: an ordinary decode step.
                    self.assertTrue(cache.resize(1, 0))
                    # Two is already a step that writes two tokens.
                    with self.assertRaisesRegex(RuntimeError, "can hold in flight"):
                        cache.resize(3, 1)
            finally:
                cache.close()
        finally:
            manager.shutdown()

    def test_a_bulk_history_jump_writes_nothing_back(self):
        """How a resumed request arrives: its whole prompt becomes history in one step.

        Nothing is written back, because nothing was ever in flight -- those units reached
        their host pages directly rather than through a cell. The write-back must therefore
        be bounded by the capacity the resize started from, not by how far history moved,
        which is a span no reservation could cover.
        """
        manager = self._manager(max_cached_units=4)
        cache = None
        try:
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                self.assertTrue(cache.resume(cast(CudaStream, stream.handle)))
                # Far more blocks than the cache could hold in flight, and no Batch at all.
                # Both are the point: nothing was ever in flight, so no write-back is owed,
                # and resize() may take the move without one. That is what lets a resumed
                # request be seeded before it joins a Batch -- which it must be, since a
                # Batch will not accept a history beyond the capacity its row already had.
                self.assertTrue(cache.resize(4096, 4096))
                self.assertEqual(cache.history_length, 4096)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_cache_state_starts_free(self):
        manager = self._manager(max_cached_units=4)
        try:
            lc_id = manager.get_layer_group_id(0)
            buffers = manager.get_sparse_cache_buffer_count(0)
            # max_cached_units counts every cell, so 4 units at expansion 1 is 4 slots and
            # 4 entries per buffer.
            entries = manager.get_sparse_cache_max_units(0)
            self.assertEqual(buffers, 1)
            self.assertEqual(entries, 4)

            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                # The lease comes with membership, so the cache state exists from the add.
                batch.add_kv_cache(cache)
                ordinals = _introspection.sparse_cache_resident_ordinals(cache, lc_id)
                self.assertEqual(len(ordinals), buffers * entries)
                self.assertTrue(all(o == BAD_PAGE_INDEX for o in ordinals), ordinals)
                batch.remove_kv_cache(cache)
                cache.close()
        finally:
            manager.shutdown()

    def test_cache_state_is_dropped_when_the_member_leaves(self):
        manager = self._manager(max_cached_units=4)
        try:
            lc_id = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertNotEqual(_introspection.sparse_cache_resident_ordinals(cache, lc_id), [])
                batch.remove_kv_cache(cache)
                # The metadata describes cells the request no longer leases, so it goes too.
                self.assertEqual(_introspection.sparse_cache_resident_ordinals(cache, lc_id), [])
                batch.add_kv_cache(cache)
                self.assertNotEqual(_introspection.sparse_cache_resident_ordinals(cache, lc_id), [])
                batch.remove_kv_cache(cache)
                cache.close()
        finally:
            manager.shutdown()

    def test_move_keeps_the_lease_that_leaving_would_drop(self):
        """The whole reason move() exists rather than remove_row() + add_kv_cache().

        That pair hands the arena lease back, and a lease comes back with every entry free,
        so it discards residency the member is still entitled to -- which is exactly what
        the test above pins. A move is not a departure, so the cells, and the base they
        start at, are the same ones afterwards.
        """
        manager = self._manager(max_cached_units=4)
        try:
            lc_id = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=3,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                self.assertEqual(batch.add_kv_cache(cache, 2), 2)
                before = _introspection.sparse_cache_resident_ordinals(cache, lc_id)
                base = _introspection.sparse_cache_base_cell(cache, lc_id)
                self.assertNotEqual(before, [])

                batch.move(0, 2)

                self.assertEqual(batch.row_of(cache), 0)
                self.assertEqual(
                    _introspection.sparse_cache_resident_ordinals(cache, lc_id),
                    before,
                    "a move is not a departure, so the cache state stays",
                )
                self.assertEqual(
                    _introspection.sparse_cache_base_cell(cache, lc_id),
                    base,
                    "and it is the same lease, not a fresh one taken from the arena",
                )
                self.assertEqual(batch.num_kv_caches, 1, "one lease held throughout")
                batch.remove_kv_cache(cache)
                cache.close()
        finally:
            manager.shutdown()

    @staticmethod
    def _hot_slots(manager: KVCacheManager) -> int:
        return sum(s.total for s in manager.get_storage_statistics(CacheLevel(0)))

    @classmethod
    def _sized_manager(cls, gpu_mb: int, capacity: int, sparse_bytes: int = 4096) -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=128,
                cache_tiers=[GpuCacheTierConfig(gpu_mb << 20), HostCacheTierConfig(64 << 20)],
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=capacity, history_length=0)] * 4)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                "key",
                                sparse_bytes,
                                is_sparse=True,
                            ),
                            # A pool group is sized from its device-backed pools, so a sparse layer
                            # carries one. This is also the shape a model uses: an indexer and the
                            # data it selects over are one attention computation.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=2, max_cached_units=4, unit_tokens=128
                        ),
                    )
                ],
            )
        )

    def test_a_sparse_pool_does_not_consume_the_gpu_quota(self):
        """A level's quota measures one medium, and these pages are not made of it.

        A slot spans every pool of its group, so the row count comes from the pools the
        quota does measure. Widening the sparse pool therefore buys more host bytes per
        row without costing a device byte, which is the point of putting history on the
        host at all.
        """
        narrow = self._sized_manager(gpu_mb=64, capacity=8192, sparse_bytes=4096)
        try:
            wide = self._sized_manager(gpu_mb=64, capacity=8192, sparse_bytes=65536)
            try:
                self.assertEqual(
                    self._hot_slots(narrow),
                    self._hot_slots(wide),
                    "a sparse pool's bytes must not be charged to the GPU quota",
                )
            finally:
                wide.shutdown()
        finally:
            narrow.shutdown()

    def test_the_slot_count_still_meets_the_declared_floor(self):
        """The constraints' floor is a minimum the quota is raised to meet."""
        manager = self._sized_manager(gpu_mb=1, capacity=8192)
        try:
            self.assertGreaterEqual(self._hot_slots(manager), 8192 // 128)
        finally:
            manager.shutdown()

    def test_the_arena_and_the_pages_have_different_bases(self):
        """Two allocations, two accessors.

        get_mem_pool_base_address keeps naming the buffer's pages whether or not it is
        sparse, which is what the transceiver and the KV transfer path address. The arena
        is where the fetch's indices point, and it is a different allocation entirely.
        """
        manager = self._manager(max_cached_units=4)
        try:
            pages = manager.get_mem_pool_base_address(LayerId(0), "key")
            with TemporaryCudaStream([]) as stream:
                batch = Batch(
                    manager,
                    cast(CudaStream, stream.handle),
                    max_batch_size=1,
                    max_blocks_per_seq=1,
                    index_tables=manager.sparse_index_table_specs(),
                )
                arena = batch.get_sparse_cache_base_address(LayerId(0), "key")
                self.assertNotEqual(int(pages), int(arena))
                self.assertNotEqual(int(arena), 0)
        finally:
            manager.shutdown()

    def test_the_arena_accessor_rejects_a_buffer_without_a_fetch_path(self):
        manager = self._manager(max_cached_units=0)
        try:
            with TemporaryCudaStream([]) as stream:
                batch = Batch(
                    manager,
                    cast(CudaStream, stream.handle),
                    max_batch_size=1,
                    max_blocks_per_seq=1,
                    index_tables=manager.sparse_index_table_specs(),
                )
                with self.assertRaises(RuntimeError):
                    batch.get_sparse_cache_base_address(LayerId(0), "key")
        finally:
            manager.shutdown()

    def test_leases_are_distinct_and_within_the_arena(self):
        """Two live requests must not be handed the same cells."""
        manager = self._manager(max_cached_units=4)
        caches = []
        try:
            lc_id = manager.get_layer_group_id(0)
            cells_per_lease = manager.get_sparse_cache_buffer_count(0) * (
                manager.get_sparse_cache_max_units(0) // manager.get_sparse_cache_units_per_page(0)
            )
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=2,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                for _ in range(2):
                    cache = manager.create_kv_cache()
                    caches.append(cache)
                    self.assertTrue(cache.resume(handle))
                    batch.add_kv_cache(cache)
                bases = [_introspection.sparse_cache_base_cell(c, lc_id) for c in caches]
                self.assertTrue(all(b >= 0 for b in bases), bases)
                self.assertEqual(len(set(bases)), len(bases), "leases must not overlap")
                for base in bases:
                    self.assertEqual(
                        base % cells_per_lease, 0, "a lease starts on a block boundary"
                    )
                for cache in caches:
                    batch.remove_kv_cache(cache)
        finally:
            for cache in caches:
                cache.close()
            manager.shutdown()

    def test_a_lease_is_returned_when_the_member_leaves(self):
        manager = self._manager(max_cached_units=4)
        try:
            lc_id = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            try:
                with TemporaryCudaStream([]) as stream:
                    handle = cast(CudaStream, stream.handle)
                    batch = Batch(
                        manager,
                        handle,
                        max_batch_size=1,
                        max_blocks_per_seq=4,
                        index_tables=manager.sparse_index_table_specs(),
                    )
                    self.assertTrue(cache.resume(handle))
                    # Resuming alone takes none: a request that is not being stepped cannot
                    # fetch, and a resumed one may still be receiving its history.
                    self.assertIsNone(_introspection.sparse_cache_base_cell(cache, lc_id))
                    batch.add_kv_cache(cache)
                    self.assertIsNotNone(_introspection.sparse_cache_base_cell(cache, lc_id))
                    batch.remove_kv_cache(cache)
                    self.assertIsNone(_introspection.sparse_cache_base_cell(cache, lc_id))
                    # Returned rather than leaked: the same cells are available again.
                    batch.add_kv_cache(cache)
                    self.assertIsNotNone(_introspection.sparse_cache_base_cell(cache, lc_id))
                    batch.remove_kv_cache(cache)
            finally:
                cache.close()
        finally:
            manager.shutdown()

    def test_the_watermark_can_cross_a_unit_boundary(self):
        """resize() must take the long path and write back, not take the shortcut.

        Nothing is resident until the fetch path is wired, so the write-back list comes
        back empty; what this covers is that the shortcut declines, the addressing is
        assembled without error, and the launches are accepted.
        """
        manager = self._manager(max_cached_units=4)
        cache = None
        batch = None
        try:
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                # Driven through a Batch because the write-back reads the converted indices
                # a publish produces, and Batch::resize publishes at the end of every call.
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                lc_id = manager.get_layer_group_id(0)
                # tokens_per_block is 128 and the unit is a whole block. Reach the end of
                # unit 0 with the last token still in flight, so that the next step's
                # watermark move genuinely carries a unit from in-flight to history --
                # which is the only way a write-back has anything to consider. Seeded
                # before joining, since a Batch only accepts a history its row already had
                # the capacity for; no unit is crossed getting here, so nothing is written
                # back yet and no published table is needed.
                self.assertTrue(cache.resize(128, 127))
                batch.add_kv_cache(cache)
                # The crossing below writes back block 0, so its converted index has to
                # exist by then.
                batch.publish()
                self.assertEqual(
                    _introspection.sparse_flush_counts(cache, lc_id),
                    [-1],
                    "no write-back has run for this row yet",
                )
                # Before the move, not after: the unit has to reach its host page while it
                # is still below capacity and above history.
                batch.flush_sparse_units([128])
                self.assertEqual(list(batch.resize([129], [128])), [1])
                self.assertEqual(cache.history_length, 128)
                self.assertEqual(
                    _introspection.sparse_flush_counts(cache, lc_id),
                    [0],
                    "a write-back ran and found nothing resident to move",
                )
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_one_flush_covers_every_row_that_moved(self):
        """The write-back is the step's, not a request's: one call serves the batch.

        Rows move independently, and a row standing still must not be given a stale range
        from whichever row did move.
        """
        manager = self._manager(max_cached_units=4)
        caches = []
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=3,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                lc_id = manager.get_layer_group_id(0)
                for _ in range(2):
                    cache = manager.create_kv_cache()
                    caches.append(cache)
                    self.assertTrue(cache.resume(handle))
                    self.assertTrue(cache.resize(128, 127))
                    batch.add_kv_cache(cache)
                batch.publish()

                # Row 0 crosses unit 0; row 1 stays put; row 2 is free.
                batch.flush_sparse_units([128, 127, 0])
                self.assertEqual(
                    _introspection.sparse_flush_counts(caches[0], lc_id),
                    [0],
                    "the row that moved was scanned",
                )
                self.assertEqual(
                    _introspection.sparse_flush_counts(caches[1], lc_id),
                    [0],
                    "a row that did not move reports none rather than keeping a stale count",
                )

                # Only the row the flush covered may advance.
                self.assertEqual(list(batch.resize([129, 128, None], [128, 127, None])), [1, 1, 0])
                with self.assertRaisesRegex(Exception, "written back"):
                    caches[1].resize(129, 128)
        finally:
            for cache in caches:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_a_move_inside_one_unit_still_takes_the_shortcut(self):
        manager = self._manager(max_cached_units=4)
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                lc_id = manager.get_layer_group_id(0)
                self.assertTrue(cache.resize(64, 63))
                batch.add_kv_cache(cache)
                batch.publish()

                batch.flush_sparse_units([64])
                self.assertEqual(list(batch.resize([64], [64])), [1])
                self.assertEqual(cache.history_length, 64)
                # Still -1: no unit boundary was crossed, so there was nothing to write back
                # and the shortcut was free to take the move.
                self.assertEqual(_introspection.sparse_flush_counts(cache, lc_id), [-1])
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_unconfigured_life_cycle_has_no_cache_state(self):
        manager = self._manager(max_cached_units=0)
        try:
            lc_id = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                self.assertTrue(cache.resume(cast(CudaStream, stream.handle)))
                self.assertEqual(_introspection.sparse_cache_resident_ordinals(cache, lc_id), [])
                cache.close()
        finally:
            manager.shutdown()


class TestSparseResolve(unittest.TestCase):
    """The sparse resolve kernel: probe index, victim choice and recency rebuild."""

    def setUp(self) -> None:
        init_cuda_once()

    @staticmethod
    def _free(count: int) -> list[int]:
        return [BAD_PAGE_INDEX] * count

    def _resolve(
        self,
        resident: list[int],
        selected: list[int],
        *,
        num_layers: int = 1,
        lru: list[int] | None = None,
        history_units: int = 1 << 20,
        capacity_units: int | None = None,
        max_in_flight_units: int = 0,
        num_rows: int = 1,
        inactive_rows: list[int] | None = None,
    ):
        entries = len(resident) // (num_layers * num_rows)
        width = len(selected) // (num_layers * num_rows)
        if lru is None:
            lru = list(range(entries)) * num_layers * num_rows
        history = [history_units] * num_rows
        capacity = [history_units if capacity_units is None else capacity_units] * num_rows
        return _introspection.sparse_resolve(
            resident,
            lru,
            num_layers,
            entries,
            selected,
            width,
            history,
            capacity,
            max_in_flight_units,
            inactive_rows,
        )

    def test_an_empty_cache_misses_everything(self):
        result = self._resolve(self._free(8), [5, 3, 7])
        self.assertEqual(result.miss_count, [3])
        # Every selection got a distinct entry, and the entries now hold those units.
        self.assertEqual(len(set(result.out_slot)), 3)
        for ordinal, slot in zip([5, 3, 7], result.out_slot):
            self.assertEqual(result.resident_ordinal[slot], ordinal)

    def test_resident_units_hit(self):
        resident = [10, 11, 12] + self._free(5)
        result = self._resolve(resident, [11, 10])
        self.assertEqual(result.miss_count, [0])
        self.assertEqual(result.out_slot, [1, 0])
        # A hit leaves the resident set untouched.
        self.assertEqual(result.resident_ordinal, resident)

    def test_padding_resolves_to_no_slot(self):
        result = self._resolve(self._free(8), [4, -1, -1])
        self.assertEqual(result.miss_count, [1])
        self.assertEqual(result.out_slot[1:], [-1, -1])

    def test_hits_outrank_units_fetched_this_step(self):
        # Entry 0 holds unit 10 and is selected; unit 99 is new.
        resident = [10] + self._free(7)
        result = self._resolve(resident, [10, 99])
        fetched = result.out_slot[1]
        self.assertEqual(result.lru_order[0], 0, "the hit should be most recently used")
        self.assertEqual(result.lru_order[1], fetched, "the fetch ranks below the hit")

    def test_victims_come_from_the_lru_tail(self):
        # A full cache, with entry 2 placed last in recency so it is the victim.
        resident = [10, 11, 12, 13]
        lru = [0, 1, 3, 2]
        result = self._resolve(resident, [99], lru=lru)
        self.assertEqual(result.miss_count, [1])
        self.assertEqual(result.out_slot, [2])
        self.assertEqual(result.resident_ordinal[2], 99)
        # The units that were not selected keep their data.
        self.assertEqual([result.resident_ordinal[i] for i in (0, 1, 3)], [10, 11, 13])

    def test_in_flight_entries_are_never_evicted(self):
        # Unit 12 is at or above the watermark, so it holds the only copy of its data.
        resident = [10, 11, 12, 13]
        lru = [0, 1, 3, 2]  # entry 2 (unit 12) is least recently used
        result = self._resolve(resident, [99], lru=lru, history_units=12)
        self.assertEqual(result.resident_ordinal[2], 12, "an in-flight unit must survive")
        self.assertEqual(result.resident_ordinal[result.out_slot[0]], 99)

    def test_in_flight_units_become_resident_without_a_fetch(self):
        # Units 10 and 11 are in flight: the model is about to write them, so they need
        # entries but their host pages hold nothing to copy.
        result = self._resolve(
            self._free(4),
            [-1],
            history_units=10,
            capacity_units=12,
            max_in_flight_units=2,
        )
        self.assertEqual(result.miss_count, [0], "an in-flight unit is never fetched")
        slots = result.in_flight_slot
        self.assertEqual(len(set(slots)), 2, "each in-flight unit got its own entry")
        self.assertEqual(result.resident_ordinal[slots[0]], 10)
        self.assertEqual(result.resident_ordinal[slots[1]], 11)

    def test_an_in_flight_unit_keeps_its_entry_across_steps(self):
        # Unit 10 is already resident in entry 2 and still in flight.
        resident = [1, 2, 10, 3]
        result = self._resolve(
            resident,
            [-1],
            history_units=10,
            capacity_units=11,
            max_in_flight_units=1,
        )
        self.assertEqual(result.in_flight_slot, [2], "a resident in-flight unit hits")
        self.assertEqual(result.miss_count, [0])

    def test_in_flight_units_are_resolved_before_the_selection(self):
        # Two entries, one in-flight unit and one selected history unit: both fit, and the
        # in-flight unit must not be the one that loses.
        result = self._resolve(
            self._free(2),
            [5],
            history_units=10,
            capacity_units=11,
            max_in_flight_units=1,
        )
        in_flight = result.in_flight_slot[0]
        selected = result.out_slot[0]
        self.assertNotEqual(in_flight, selected)
        self.assertEqual(result.resident_ordinal[in_flight], 10)
        self.assertEqual(result.resident_ordinal[selected], 5)
        self.assertEqual(result.miss_count, [1], "only the history unit is fetched")

    def test_rows_resolve_independently(self):
        """Each row reads its own cache, so one row's misses must not disturb another's."""
        # Row 0 holds unit 7; row 1 is empty. Both select 7.
        resident = [7, BAD_PAGE_INDEX] + [BAD_PAGE_INDEX, BAD_PAGE_INDEX]
        result = self._resolve(resident, [7, 7], num_rows=2)
        self.assertEqual(result.miss_count, [0, 1], "row 0 hits, row 1 fetches")
        self.assertEqual(result.out_slot[0], 0, "row 0 reuses its own entry")
        # Row 1's unit landed in row 1's own cache, leaving row 0 untouched.
        self.assertEqual(result.resident_ordinal[:2], [7, BAD_PAGE_INDEX])
        self.assertEqual(result.resident_ordinal[2 + result.out_slot[1]], 7)

    def test_misses_compact_across_rows(self):
        """Every row's misses land in one dense list, each entry naming its own row."""
        # Three rows of two entries each; rows 0 and 2 miss, row 1 hits what it selects.
        resident = (
            [BAD_PAGE_INDEX, BAD_PAGE_INDEX]
            + [4, BAD_PAGE_INDEX]
            + [BAD_PAGE_INDEX, BAD_PAGE_INDEX]
        )
        result = self._resolve(resident, [3, 4, 5], num_rows=3, history_units=8)
        self.assertEqual(result.miss_count, [1, 0, 1], "row 1 hits its selection")
        self.assertEqual(
            result.miss_total, [2], "the list holds exactly the misses, not a slot per row"
        )

        # Position says nothing about which row an entry came from, and the CTAs race, so
        # the list is only defined as a set.
        entries = {
            (result.miss_row[i], result.miss_ordinal[i]) for i in range(result.miss_total[0])
        }
        self.assertEqual(entries, {(0, 3), (2, 5)})
        for i in range(result.miss_total[0]):
            row = result.miss_row[i]
            slot = result.miss_slot[i]
            self.assertEqual(
                result.resident_ordinal[row * 2 + slot],
                result.miss_ordinal[i],
                "the entry names the cell the unit was given",
            )

    def test_a_row_holding_no_reservation_is_skipped(self):
        """A free row, or one that has suspended, stages a default view the kernel skips."""
        resident = [BAD_PAGE_INDEX] * 4
        result = self._resolve(resident, [5, 5], num_rows=2, inactive_rows=[1])
        # Row 0 resolved; row 1 was left exactly as the caller initialised it.
        self.assertEqual(result.miss_count[0], 1)
        self.assertEqual(result.miss_count[1], 0, "an inactive row does no work")
        self.assertEqual(result.resident_ordinal[2:], [BAD_PAGE_INDEX] * 2)

    def test_layers_resolve_independently(self):
        # Layer 0 hits, layer 1 misses.
        resident = [7, BAD_PAGE_INDEX] + self._free(2)
        selected = [7, -1, 42, -1]
        result = self._resolve(resident, selected, num_layers=2)
        self.assertEqual(result.miss_count, [0, 1])
        self.assertEqual(result.out_slot[0], 0)
        self.assertEqual(result.resident_ordinal[2 + result.out_slot[2]], 42)

    def test_large_caches_preserve_victims_and_recency_order(self):
        """Hits, fetched victims, and untouched entries retain their relative order."""
        import cuda.bindings.runtime as cudart

        # The 8192-entry case needs 147552 bytes of dynamic shared memory per resolve block,
        # on top of what the kernel declares statically.
        err, device = cudart.cudaGetDevice()
        self.assertEqual(int(err), int(cudart.cudaError_t.cudaSuccess))
        err, shared_optin = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, device
        )
        self.assertEqual(int(err), int(cudart.cudaError_t.cudaSuccess))
        if shared_optin < 160 << 10:
            self.skipTest(
                f"needs 160 KiB of shared memory per block; this device allows {shared_optin}"
            )
        for entries in (31, 32, 33, 255, 256, 257, 2048, 4096, 4224, 8192):
            with self.subTest(entries=entries):
                rng = random.Random(entries)
                history = 100000
                in_flight = 4
                resident = list(range(entries - in_flight)) + list(
                    range(history, history + in_flight)
                )
                resident[: entries // 8] = self._free(entries // 8)
                rng.shuffle(resident)
                lru = list(range(entries))
                rng.shuffle(lru)
                count = min(entries // 4, 512)
                hits = rng.sample(
                    [ordinal for ordinal in resident if 0 <= ordinal < history], count
                )
                misses = list(range(50000, 50000 + count))
                selected = hits + misses + [-1] * 7
                rng.shuffle(selected)
                hit_slots = {
                    slot
                    for slot, ordinal in enumerate(resident)
                    if ordinal in hits or history <= ordinal < history + in_flight
                }
                eligible = [slot for slot in lru if slot not in hit_slots]
                victims = set(eligible[-count:])
                expected_lru = (
                    [slot for slot in lru if slot in hit_slots]
                    + [slot for slot in lru if slot in victims]
                    + [slot for slot in lru if slot not in hit_slots | victims]
                )
                try:
                    result = self._resolve(
                        resident,
                        selected,
                        lru=lru,
                        history_units=history,
                        capacity_units=history + in_flight,
                        max_in_flight_units=in_flight,
                    )
                except ValueError as error:
                    if "above this device's limit" in str(error):
                        raise unittest.SkipTest(str(error)) from error
                    raise
                self.assertEqual(result.miss_count, [count])
                self.assertEqual(result.lru_order, expected_lru)
                fetched_slots = set()
                for ordinal, slot in zip(selected, result.out_slot):
                    if ordinal < 0:
                        self.assertEqual(slot, -1)
                    else:
                        self.assertEqual(result.resident_ordinal[slot], ordinal)
                        if ordinal in misses:
                            fetched_slots.add(slot)
                self.assertEqual(fetched_slots, victims)
                for slot, ordinal in enumerate(resident):
                    if slot not in victims:
                        self.assertEqual(result.resident_ordinal[slot], ordinal)
                self.assertEqual(
                    result.in_flight_slot,
                    [resident.index(history + i) for i in range(in_flight)],
                )

    def test_colliding_selections_preserve_slots_and_recency(self) -> None:
        """Sparse ordinals and padding preserve the same eviction and stable-order contract."""
        for entries in (63, 257, 2080):
            with self.subTest(entries=entries):
                history = 1 << 28
                resident = [i * 2048 for i in range(entries - 2)] + [history, history + 1]
                lru = list(range(entries))
                random.Random(entries).shuffle(lru)
                count = min(entries // 4, 32)
                hits = resident[:count]
                misses = [(entries + i) * 2048 for i in range(count)]
                selected = [-1, *misses, -1, *reversed(hits), -1]
                hit_slots = set(range(count)) | {entries - 2, entries - 1}
                eligible = [slot for slot in lru if slot not in hit_slots]
                victims = set(eligible[-count:])
                expected_lru = (
                    [slot for slot in lru if slot in hit_slots]
                    + [slot for slot in lru if slot in victims]
                    + [slot for slot in lru if slot not in hit_slots | victims]
                )
                result = self._resolve(
                    resident,
                    selected,
                    lru=lru,
                    history_units=history,
                    capacity_units=history + 2,
                    max_in_flight_units=4,
                )
                self.assertEqual(result.lru_order, expected_lru)
                self.assertEqual(result.miss_count, [count])
                self.assertEqual(result.miss_total, [count])
                self.assertEqual(result.in_flight_slot, [entries - 2, entries - 1, -1, -1])
                for ordinal, slot in zip(selected, result.out_slot):
                    if ordinal < 0:
                        self.assertEqual(slot, -1)
                    else:
                        self.assertEqual(result.resident_ordinal[slot], ordinal)
                        if ordinal in hits:
                            self.assertEqual(slot, resident.index(ordinal))
                for slot in set(range(entries)) - victims:
                    self.assertEqual(result.resident_ordinal[slot], resident[slot])


class TestSparseUnitCopy(unittest.TestCase):
    """Moving a unit between its cold page and its cache cell, in both directions."""

    def setUp(self) -> None:
        init_cuda_once()

    @staticmethod
    def _pattern(size: int) -> list[int]:
        return [(i * 7 + 13) % 251 for i in range(size)]

    def test_a_contiguous_unit_lands_at_its_entry(self):
        # One layer, token-major: a unit is one contiguous run of bytes.
        unit, units_per_page, pages = 32, 2, 3
        page_stride = unit * units_per_page
        host = self._pattern(page_stride * pages)
        # Block 2 lives in host page 0, and its unit 1 is selected.
        page_table = [9, 9, 0]
        ordinal = 2 * units_per_page + 1
        result = _introspection.sparse_unit_copy(
            host_pool=host,
            host_page_stride=page_stride,
            page_table=page_table,
            device_arena=[0] * (page_stride * 2),
            # A non-zero lease base: entry 0 belongs to the second cell of the arena, which
            # is what every request past the first gets.
            base_cell=1,
            num_blocks=len(page_table),
            cells_per_buffer=1,
            buffer_bytes=page_stride,
            num_chunks=1,
            chunk_bytes=unit,
            chunk_stride=0,
            unit_stride=unit,
            expansion=units_per_page,
            miss_ordinal=[ordinal],
            miss_slot=[0],
            miss_count=[1],
            selection_width=1,
            buffer_layer=list(range(1)),
        )
        # Entry 0 of a lease based at cell 1 is the arena's cell 1, unit 0.
        expected = host[unit : unit * 2]
        self.assertEqual(result[page_stride : page_stride + unit], expected)
        # Nothing else was written.
        self.assertEqual(result[:page_stride], [0] * page_stride)

    def test_chunked_layout_follows_the_strides(self):
        # Two heads: a unit is two runs, chunk_stride apart on both sides.
        chunk, chunks, units = 16, 2, 2
        chunk_stride = chunk * units
        page_stride = chunk_stride * chunks
        host = self._pattern(page_stride)
        result = _introspection.sparse_unit_copy(
            host_pool=host,
            host_page_stride=page_stride,
            page_table=[0],
            device_arena=[0] * (page_stride),
            base_cell=0,
            num_blocks=1,
            cells_per_buffer=1,
            buffer_bytes=page_stride,
            num_chunks=chunks,
            chunk_bytes=chunk,
            chunk_stride=chunk_stride,
            unit_stride=chunk,
            expansion=units,
            miss_ordinal=[1],
            miss_slot=[1],
            miss_count=[1],
            selection_width=1,
            buffer_layer=list(range(1)),
        )
        # Unit 1 of each chunk moves to unit 1 of the same chunk, verbatim.
        for c in range(chunks):
            base = c * chunk_stride + chunk
            self.assertEqual(result[base : base + chunk], host[base : base + chunk])
        # Unit 0 of each chunk was not selected.
        for c in range(chunks):
            base = c * chunk_stride
            self.assertEqual(result[base : base + chunk], [0] * chunk)

    def test_layers_land_in_their_own_sub_slot(self):
        # A cold page concatenates two layers' buffers; the cache cell is derived.
        unit, layers = 16, 2
        host = self._pattern(unit * layers)
        result = _introspection.sparse_unit_copy(
            host_pool=host,
            # One buffer, not the coalesced page: a converted index counts whole buffers.
            host_page_stride=unit,
            # Deliberately crossed, which is now the table's doing rather than a separate
            # offset array: slot 0 of a pool coalescing two buffers converts to index 0 or
            # 1, and layer 0 is given the second.
            page_table=[1, 0],
            device_arena=[0] * (unit * layers),
            base_cell=0,
            num_blocks=1,
            cells_per_buffer=1,
            buffer_bytes=unit,
            num_chunks=1,
            chunk_bytes=unit,
            chunk_stride=0,
            unit_stride=unit,
            expansion=1,
            miss_ordinal=[0, 0],
            miss_slot=[0, 0],
            miss_count=[1, 1],
            selection_width=1,
            buffer_layer=list(range(layers)),
        )
        # The cold page comes from the entry and the hot cell is arithmetic on the sub-slot,
        # so layer 0 reads the second cold buffer into cell 0, and layer 1 the reverse.
        self.assertEqual(result[:unit], host[unit:])
        self.assertEqual(result[unit:], host[:unit])

    def test_a_layer_without_misses_copies_nothing(self):
        unit = 16
        host = self._pattern(unit)
        result = _introspection.sparse_unit_copy(
            host_pool=host,
            host_page_stride=unit,
            page_table=[0],
            device_arena=[0] * (unit),
            base_cell=0,
            num_blocks=1,
            cells_per_buffer=1,
            buffer_bytes=unit,
            num_chunks=1,
            chunk_bytes=unit,
            chunk_stride=0,
            unit_stride=unit,
            expansion=1,
            miss_ordinal=[0],
            miss_slot=[0],
            miss_count=[0],
            selection_width=1,
            buffer_layer=list(range(1)),
        )
        self.assertEqual(result, [0] * unit)

    def test_an_unaligned_unit_still_copies(self):
        # 20 bytes is not a multiple of 16, so the wide path must not be taken.
        unit = 20
        host = self._pattern(unit * 2)
        result = _introspection.sparse_unit_copy(
            host_pool=host,
            host_page_stride=unit * 2,
            page_table=[0],
            device_arena=[0] * (unit * 2),
            base_cell=0,
            num_blocks=1,
            cells_per_buffer=1,
            buffer_bytes=unit * 2,
            num_chunks=1,
            chunk_bytes=unit,
            chunk_stride=0,
            unit_stride=unit,
            expansion=2,
            miss_ordinal=[1],
            miss_slot=[1],
            miss_count=[1],
            selection_width=1,
            buffer_layer=list(range(1)),
        )
        self.assertEqual(result[unit:], host[unit:])

    def test_the_work_splits_across_the_grid(self):
        """Giving the copy more CTAs only divides the work list, never changes the result."""
        unit, units_per_page, misses = 16, 8, 5
        page_stride = unit * units_per_page
        host = self._pattern(page_stride)
        # More warps than work on the last pass: the surplus must find nothing to do.
        for grid_ctas in (1, 2, 4, 8):
            with self.subTest(grid_ctas=grid_ctas):
                result = _introspection.sparse_unit_copy(
                    host_pool=host,
                    host_page_stride=page_stride,
                    page_table=[0],
                    device_arena=[0] * page_stride,
                    base_cell=0,
                    num_blocks=1,
                    cells_per_buffer=1,
                    buffer_bytes=page_stride,
                    num_chunks=1,
                    chunk_bytes=unit,
                    chunk_stride=0,
                    unit_stride=unit,
                    expansion=units_per_page,
                    miss_ordinal=list(range(misses)),
                    miss_slot=list(range(misses)),
                    miss_count=[misses],
                    selection_width=misses,
                    buffer_layer=list(range(1)),
                    grid_ctas=grid_ctas,
                )
                self.assertEqual(result[: unit * misses], host[: unit * misses])
                self.assertEqual(result[unit * misses :], [0] * (page_stride - unit * misses))


class TestSparseFetch(unittest.TestCase):
    """KVCacheManager.fetch_sparse_pages: one layer's selections, resolved for a batch."""

    TOKENS_PER_BLOCK = 128
    SELECTED = 2
    CACHED_UNITS = 4

    def setUp(self) -> None:
        init_cuda_once()

    # Bytes per sparse buffer. A pool holds buffers of one size, so giving two roles a
    # different size is how a test puts them in a pool of their own.
    SPARSE_BYTES = 4096

    @classmethod
    def _manager(
        cls,
        roles: tuple[str, ...] = ("key", "value"),
        sizes: dict[str, int] | None = None,
    ) -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=cls.TOKENS_PER_BLOCK,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                # The cache arena is reserved up front, so a sparse deployment must say
                # how many requests run together.
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 4)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                role,
                                (sizes or {}).get(role, cls.SPARSE_BYTES),
                                is_sparse=True,
                            )
                            for role in roles
                        ]
                        + [
                            # A pool group sizes its slots from its device-backed pools, so a
                            # sparse layer carries one -- the shape a model uses anyway.
                            BufferConfig("index", 256)
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=cls.SELECTED,
                            max_cached_units=cls.CACHED_UNITS,
                            unit_tokens=cls.TOKENS_PER_BLOCK,
                        ),
                    )
                ],
            )
        )

    @staticmethod
    def _buffers(roles: tuple[str, ...]) -> list[BufferId]:
        return [BufferId(0, role) for role in roles]

    def _torch(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to pass device arrays")
        return torch

    @contextlib.contextmanager
    def _fixture(
        self,
        roles: tuple[str, ...] = ("key", "value"),
        rows: int = 2,
        blocks: int = 4,
        sizes: dict[str, int] | None = None,
    ):
        """A resumed member in row 0 of a two-row batch, with history to select from."""
        torch = self._torch()
        manager = self._manager(roles, sizes)
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=rows,
                    max_blocks_per_seq=blocks,
                    index_tables=manager.sparse_index_table_specs(),
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                # Two whole blocks of history, so ordinals 0 and 1 are eligible.
                tokens = 2 * self.TOKENS_PER_BLOCK
                # Seeded before joining: a Batch only accepts a history its row already had
                # the capacity for, and this one is jumping straight past both.
                self.assertTrue(cache.resize(tokens, tokens))
                batch.add_kv_cache(cache)
                # resize() publishes, so the batch is already current here.
                self.assertEqual(batch.resize([tokens, None], [tokens, None]), [True, False])
                yield torch, manager, batch, cache, handle
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def _out(self, torch, rows: int, num_buffers: int, width: int | None = None):
        """Out is compact: one index per selection slot, not per unit ordinal."""
        return torch.full(
            (rows, num_buffers, self.SELECTED if width is None else width),
            12345,
            dtype=torch.int32,
            device="cuda",
        )

    def test_every_selection_slot_is_written(self):
        """A stale index must never survive, so padded slots take the sentinel too.

        Of the slots a buffer maps to. Buffers whose indices coincide share one, and the
        entries of ``out`` no buffer maps to are not written at all -- see
        ``sparse_out_slots``. These two buffers do not coincide, so every entry is live.
        """
        roles = ("key", "value")
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            selected = torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32, device="cuda")
            out = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), selected, out, handle)
            torch.cuda.synchronize()

            # Slot 0 carried a selection and resolved for both buffers of the member's row.
            self.assertTrue(bool((out[0, :, 0] != BAD_PAGE_INDEX).all()))
            # Slot 1 was padded, so it reads as absent rather than as the fill we started from.
            self.assertTrue(bool((out[0, :, 1] == BAD_PAGE_INDEX).all()))

    def test_mirrored_pools_resolve_a_selection_to_one_index(self):
        """Mirrored pools give a selection one index, so one slice of ``out`` serves both.

        K and V share a pool; their block scales share another. The two pools hold the same
        number of buffers, so a lease spans the same cells in each and a selection resolves
        to the same index in both. ``sparse_out_slots`` reports that as a shared slot, and
        the fetch then resolves it once.

        The equality itself is no longer observable here: the merge is what removes the
        second copy, so there is nothing left to compare it against. It is enforced instead
        where the slots are built, which refuses to share a slot unless the two buffers'
        arena geometry -- cellsPerLease, cellsPerBuffer and slotInPool -- agrees.

        Checked on the *second* member, whose lease is not zero. An index is
        ``leaseIndex * cellsPerLease + slotInPool * cellsPerBuffer + ...``, and
        ``cellsPerLease`` is the only term that differs between pools of unequal width, so
        at lease zero a wrongly shared slot would look identical to a right one.
        """
        roles = ("key", "value", "key_scale", "value_scale")
        sizes = {"key_scale": 512, "value_scale": 512}
        with self._fixture(roles, sizes=sizes) as (torch, manager, batch, _cache, handle):
            second = manager.create_kv_cache()
            try:
                self.assertTrue(second.resume(handle))
                tokens = 2 * self.TOKENS_PER_BLOCK
                self.assertTrue(second.resize(tokens, tokens))
                batch.add_kv_cache(second)
                self.assertEqual(batch.row_of(second), 1, "the second member takes row 1")
                self.assertEqual(batch.resize([tokens, tokens], [tokens, tokens]), [True, True])

                buffers = self._buffers(roles)
                self.assertEqual(batch.sparse_out_slots(buffers), [0, 1, 0, 1])

                selected = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="cuda")
                out = self._out(torch, 2, len(roles))
                batch.fetch_sparse_pages(buffers, selected, out, handle)
                torch.cuda.synchronize()

                for row in (0, 1):
                    key, value = out[row, 0, :], out[row, 1, :]
                    self.assertTrue(
                        bool((key != BAD_PAGE_INDEX).all()), f"row {row} should resolve"
                    )
                    # The two buffers of a pool sit at consecutive slots, so they differ by
                    # a constant. Stated as a difference because the stride is the arena's
                    # business, not this test's.
                    deltas = (value - key).tolist()
                    self.assertEqual(len(set(deltas)), 1, f"row {row}: one stride, {deltas}")
                    self.assertNotEqual(deltas[0], 0, f"row {row}: key and value share a cell")

                # Different leases, or the geometry term the sharing turns on is multiplied
                # away and this proves nothing.
                self.assertNotEqual(out[0, 0, :].tolist(), out[1, 0, :].tolist())
            finally:
                if batch.row_of(second) is not None:
                    batch.remove_kv_cache(second)
                second.close()

    def test_mirrored_pools_share_one_output_slice(self):
        """Buffers whose indices coincide are resolved once, into one slice of ``out``.

        The scatter runs per pool, and a pool whose slices an earlier one already wrote is
        not launched: the work and the writes are what the sharing saves. ``out`` keeps one
        entry per buffer, so the entries no buffer maps to are left as they were, which is
        what this checks -- a merged layout that still wrote them would save nothing.
        """
        roles = ("key", "value", "key_scale", "value_scale")
        sizes = {"key_scale": 512, "value_scale": 512}
        with self._fixture(roles, sizes=sizes) as (torch, manager, batch, _cache, handle):
            buffers = self._buffers(roles)
            # Mirrored pools: key with its scale, value with its scale.
            self.assertEqual(batch.sparse_out_slots(buffers), [0, 1, 0, 1])

            selected = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
            out = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(buffers, selected, out, handle)
            torch.cuda.synchronize()

            # The slots buffers map to carry real indices.
            self.assertTrue(bool((out[0, 0, :] != BAD_PAGE_INDEX).all()))
            self.assertTrue(bool((out[0, 1, :] != BAD_PAGE_INDEX).all()))
            # The rest were never written, so they still hold the fill the test started
            # from -- not the sentinel, which would mean the scatter had run over them.
            self.assertTrue(bool((out[:, 2:, :] == 12345).all()), "a shared slice was rewritten")

    def test_a_scale_shared_by_two_buffers_keeps_its_own_slice(self):
        """Coinciding is the test, not pooling: an unmirrored pool gets a slice of its own.

        One scale serving both key and value is a pool of one buffer, so its entries do not
        line up with the key-and-value pool's and nothing may be shared with it.
        """
        roles = ("key", "value", "kv_scale")
        sizes = {"kv_scale": 512}
        with self._fixture(roles, sizes=sizes) as (torch, manager, batch, _cache, handle):
            buffers = self._buffers(roles)
            self.assertEqual(batch.sparse_out_slots(buffers), [0, 1, 2])

            selected = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
            out = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(buffers, selected, out, handle)
            torch.cuda.synchronize()
            # Nothing is shared, so every entry is a slot some buffer maps to.
            self.assertTrue(bool((out[0, :, :] != BAD_PAGE_INDEX).all()))

    def test_the_hit_rate_counts_selected_units_and_nothing_else(self):
        """Hits and misses are selection entries naming real history, and only those.

        A padded entry is neither, which is why the pair cannot be derived on the host from
        the selection's width -- the padding is the model's and lives on the device.
        """
        roles = ("key",)
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            self.assertEqual(batch.peek_hit_miss_counter(), (0, 0))
            buffers = self._buffers(roles)
            out = self._out(torch, 2, len(roles))

            # Row 0 selects two real units; row 1 holds no reservation and resolves nothing.
            both = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
            batch.fetch_sparse_pages(buffers, both, out, handle)
            torch.cuda.synchronize()
            self.assertEqual(batch.peek_hit_miss_counter(), (0, 2), "a cold cache misses both")

            # The same units are resident now, so the second fetch copies nothing.
            batch.fetch_sparse_pages(buffers, both, out, handle)
            torch.cuda.synchronize()
            self.assertEqual(batch.peek_hit_miss_counter(), (2, 2))

            # One real selection and one padded: the padding moves neither figure.
            padded = torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32, device="cuda")
            batch.fetch_sparse_pages(buffers, padded, out, handle)
            torch.cuda.synchronize()
            self.assertEqual(batch.peek_hit_miss_counter(), (3, 2))

            batch.reset_hit_miss_counter()
            torch.cuda.synchronize()
            self.assertEqual(batch.peek_hit_miss_counter(), (0, 0))

    def test_a_write_back_leaves_the_hit_rate_alone(self):
        """The flush shares the copy kernel; its units moved are not cache misses."""
        roles = ("key",)
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            buffers = self._buffers(roles)
            out = self._out(torch, 2, len(roles))
            both = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
            batch.fetch_sparse_pages(buffers, both, out, handle)
            torch.cuda.synchronize()
            before = batch.peek_hit_miss_counter()

            batch.flush_sparse_units([2 * self.TOKENS_PER_BLOCK, 0])
            torch.cuda.synchronize()
            self.assertEqual(batch.peek_hit_miss_counter(), before)

    def test_the_output_follows_the_selection_order(self):
        """Slot k of out and slot k of the selection describe the same unit.

        Padding is left where the selection put it rather than packed out, which is what
        lets a caller pair the two arrays without a second index.
        """
        roles = ("key",)
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            forward = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
            first = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), forward, first, handle)

            # The same two units, named in the other order: the indices must follow.
            reverse = torch.tensor([[1, 0], [-1, -1]], dtype=torch.int32, device="cuda")
            second = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), reverse, second, handle)
            torch.cuda.synchronize()

            self.assertEqual(int(first[0, 0, 0]), int(second[0, 0, 1]))
            self.assertEqual(int(first[0, 0, 1]), int(second[0, 0, 0]))

    def test_a_row_holding_no_reservation_reads_as_absent(self):
        roles = ("key", "value")
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            selected = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="cuda")
            out = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), selected, out, handle)
            torch.cuda.synchronize()
            self.assertTrue(
                bool((out[1] == BAD_PAGE_INDEX).all()),
                "a free row selects nothing, however the caller filled its selection",
            )

    def test_each_buffer_of_a_layer_gets_its_own_index(self):
        """A layer's buffers share the resolve but not their cells.

        The cache packs one buffer's entries densely across the pages it reserved rather
        than leaving each at its nominal cell, so one selection lands somewhere different
        in every buffer. That is why out carries a buffer dimension at all.
        """
        roles = ("key", "value")
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            selected = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
            out = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), selected, out, handle)
            torch.cuda.synchronize()

            for slot in (0, 1):
                key, value = int(out[0, 0, slot]), int(out[0, 1, slot])
                self.assertNotEqual(key, BAD_PAGE_INDEX)
                self.assertNotEqual(key, value, f"selection slot {slot} shares a cell")

    def test_a_hit_resolves_to_the_same_index_as_the_fetch_that_filled_it(self):
        """A second call must reuse the entry rather than fetch it again."""
        roles = ("key",)
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            selected = torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32, device="cuda")
            first = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), selected, first, handle)
            second = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), selected, second, handle)
            torch.cuda.synchronize()
            self.assertEqual(int(first[0, 0, 0]), int(second[0, 0, 0]))

    def test_the_resolve_makes_the_selected_units_resident(self):
        """The indices alone could be anything; residency is what the fetch actually changed."""
        roles = ("key",)
        with self._fixture(roles) as (torch, manager, batch, cache, handle):
            lg_id = manager.get_layer_group_id(0)
            before = _introspection.sparse_cache_resident_ordinals(cache, lg_id)
            self.assertNotIn(0, before, "nothing is cached before the first fetch")
            self.assertNotIn(1, before)

            selected = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
            out = self._out(torch, 2, len(roles))
            batch.fetch_sparse_pages(self._buffers(roles), selected, out, handle)
            torch.cuda.synchronize()

            after = _introspection.sparse_cache_resident_ordinals(cache, lg_id)
            self.assertIn(0, after, "the selected units are resident afterwards")
            self.assertIn(1, after)

    def test_buffers_must_be_one_whole_layer(self):
        roles = ("key", "value")
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            selected = torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32, device="cuda")
            # A subset would leave the buffers it omits stale while the metadata, which the
            # resolve updates for the whole layer, claimed they were resident.
            with self.assertRaises(RuntimeError):
                batch.fetch_sparse_pages(
                    [BufferId(0, "key")], selected, self._out(torch, 2, 1), handle
                )

    def test_selected_must_carry_one_row_per_batch_row(self):
        """Per query token is what a kernel's selection looks like; this call is not that.

        A decode step gives a request one query token, so the two agree there. A prefill
        chunk would give it many, and the resolve has no way to hold the union resident.
        """
        roles = ("key",)
        with self._fixture(roles, rows=2) as (torch, manager, batch, _cache, handle):
            # Four query tokens for a two-row batch: what a prefill caller would pass.
            selected = torch.full((4, self.SELECTED), -1, dtype=torch.int32, device="cuda")
            out = self._out(torch, 4, len(roles))
            with self.assertRaises(ValueError):
                batch.fetch_sparse_pages(self._buffers(roles), selected, out, handle)

    def test_out_must_carry_one_row_per_buffer(self):
        roles = ("key", "value")
        with self._fixture(roles) as (torch, manager, batch, _cache, handle):
            selected = torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32, device="cuda")
            with self.assertRaises(ValueError):
                batch.fetch_sparse_pages(
                    self._buffers(roles), selected, self._out(torch, 2, 1), handle
                )


class TestIndexTables(unittest.TestCase):
    """Which converted index tables a Batch builds, and which specs share one.

    A spec states what a kernel must address together. Two specs share a table when
    their converted entries coincide, which needs the same life cycle, the same page index
    scale, and the same offsets -- the condition a quantised layer's data and block scales
    satisfy because their pool group mirrors their slots.
    """

    def setUp(self) -> None:
        init_cuda_once()

    @staticmethod
    def _manager(*, roles: tuple[str, ...], layers: int = 2) -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=32,
                cache_tiers=[GpuCacheTierConfig(quota=8 << 20)],
                layers=[
                    AttentionLayerConfig(
                        layer_id=LayerId(layer),
                        buffers=[BufferConfig(role=role, size=1024) for role in roles],
                    )
                    for layer in range(layers)
                ],
                constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=32, history_length=0)])],
            )
        )

    @staticmethod
    def _batch(manager, specs, max_batch_size=1):
        return Batch(
            manager,
            NULL_STREAM,
            max_batch_size=max_batch_size,
            max_blocks_per_seq=4,
            index_tables=specs,
        )

    def test_a_rows_entries_sit_together(self):
        """A row holds its entries side by side, rather than one plane per entry.

        The shape an attention kernel takes a key and value page list in: XQA's
        kvCachePageList is [batchSize][beamWidth][2][maxNbPagesPerSeq]. A single-row
        batch cannot tell the two layouts apart, so this one has several.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to read the view")
        manager = self._manager(roles=("key", "value"))
        try:
            spec = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            batch = self._batch(manager, [spec], max_batch_size=3)
            view = torch.from_dlpack(batch.converted_index_table(0))
            self.assertEqual(tuple(view.shape), (3, 2, 4))
            # One row spans every entry; one entry spans one row of blocks.
            self.assertEqual(view.stride(), (2 * 4, 4, 1))
        finally:
            manager.shutdown()

    def test_the_view_is_shaped_for_the_entries_the_table_serves(self):
        """One entry per buffer of the spec, adjacent within each row."""
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to read the view")
        manager = self._manager(roles=("key", "value"))
        try:
            spec = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            batch = self._batch(manager, [spec])
            view = torch.from_dlpack(batch.converted_index_table(0))
            # Row-major in the entries, so a row carries both buffers side by side: two
            # buffers in the spec gives [max_batch, 2, max_blocks_per_seq].
            self.assertEqual(tuple(view.shape), (1, 2, 4))
            self.assertEqual(view.dtype, torch.int32)
            self.assertTrue(view.is_cuda)
            # The entry a buffer reads is its position in the spec that named it.
            self.assertEqual(_introspection.batch_index_table_slot(batch, 0, "value")[1], 1)
            with self.assertRaises(Exception):
                batch.converted_index_table(batch.num_index_tables)
        finally:
            manager.shutdown()

    def test_a_spec_is_reachable_from_every_buffer_it_names(self):
        """Entry order follows the spec's buffer order, which is what a kernel indexes."""
        manager = self._manager(roles=("key", "value"))
        try:
            spec = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            batch = self._batch(manager, [spec])
            self.assertEqual(batch.num_index_tables, 1)
            self.assertEqual(_introspection.batch_index_table_slot(batch, 0, "key"), (0, 0))
            self.assertEqual(_introspection.batch_index_table_slot(batch, 0, "value"), (0, 1))
            # One base for the whole spec: the offsets are already in the entries.
            self.assertEqual(
                _introspection.batch_index_table_base(batch, 0, "key"),
                _introspection.batch_index_table_base(batch, 0, "value"),
            )
        finally:
            manager.shutdown()

    def test_layers_of_one_life_cycle_share_a_table_under_shared(self):
        """SHARED is exactly this: entries that every layer of the group can read."""
        manager = self._manager(roles=("key",), layers=2)
        try:
            lg = manager.get_layer_group_id(0)
            specs = [IndexTableSpec([BufferId(layer, "key")]) for layer in range(2)]
            batch = self._batch(manager, specs)
            self.assertEqual(batch.index_mode(lg), PageIndexMode.SHARED)
            self.assertEqual(batch.num_index_tables, 1)
            self.assertEqual(
                _introspection.batch_index_table_slot(batch, 0, "key")[0],
                _introspection.batch_index_table_slot(batch, 1, "key")[0],
            )
            # Two specs merged onto it, one per layer.
            self.assertEqual(len(batch.index_table(0)), 2)
        finally:
            manager.shutdown()

    def test_per_layer_keeps_the_layers_apart(self):
        """However alike the entries look, a PER_LAYER table serves one layer."""
        manager = self._manager(roles=("key",), layers=2)
        try:
            specs = [IndexTableSpec([BufferId(layer, "key")]) for layer in range(2)]
            batch = Batch(
                manager,
                NULL_STREAM,
                max_batch_size=1,
                max_blocks_per_seq=4,
                mode=PageIndexMode.PER_LAYER,
                index_tables=specs,
            )
            self.assertEqual(batch.num_index_tables, 2)
            self.assertNotEqual(
                _introspection.batch_index_table_slot(batch, 0, "key")[0],
                _introspection.batch_index_table_slot(batch, 1, "key")[0],
            )
        finally:
            manager.shutdown()

    @staticmethod
    def _quantised_manager(*, paired_scales: bool) -> KVCacheManager:
        """One layer whose data and block-scale buffers land in separate pools.

        Pools group buffers of equal size, and a pool's scale is how many buffers it
        coalesces. So two same-size scale buffers give scale 2 like the data pair, while a
        lone one gives scale 1.
        """
        scale_roles = ["k_scale", "v_scale"] if paired_scales else ["k_scale"]
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=32,
                cache_tiers=[GpuCacheTierConfig(quota=8 << 20)],
                layers=[
                    AttentionLayerConfig(
                        layer_id=LayerId(0),
                        buffers=[BufferConfig(role=role, size=1024) for role in ("key", "value")]
                        + [BufferConfig(role=role, size=256) for role in scale_roles],
                    )
                ],
                constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=32, history_length=0)])],
            )
        )

    def test_data_and_block_scales_share_a_table_when_their_scales_match(self):
        """The quantised case the merge rule exists for.

        Both pools coalesce two buffers and the pool group mirrors their slots, so the
        converted entries coincide and one table serves both -- through different bases.
        """
        manager = self._quantised_manager(paired_scales=True)
        try:
            self.assertEqual(
                manager.get_page_index_scale(0, "key"), manager.get_page_index_scale(0, "k_scale")
            )
            data = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            scales = IndexTableSpec([BufferId(0, "k_scale"), BufferId(0, "v_scale")])
            batch = self._batch(manager, [data, scales])

            self.assertEqual(batch.num_index_tables, 1)
            self.assertEqual(len(batch.index_table(0)), 2)
            # Same entries, so the same table at the same slots...
            self.assertEqual(
                _introspection.batch_index_table_slot(batch, 0, "key"),
                _introspection.batch_index_table_slot(batch, 0, "k_scale"),
            )
            self.assertEqual(
                _introspection.batch_index_table_slot(batch, 0, "value"),
                _introspection.batch_index_table_slot(batch, 0, "v_scale"),
            )
            # ...reached through different bases, which is what keeps them distinct.
            self.assertNotEqual(
                _introspection.batch_index_table_base(batch, 0, "key"),
                _introspection.batch_index_table_base(batch, 0, "k_scale"),
            )
        finally:
            manager.shutdown()

    def test_a_spec_names_its_own_table_without_going_through_a_buffer(self):
        """The spec a caller asked for is the handle it gets its table back by.

        Mirrored specs resolve to one id, which is the same statement
        ``index_table_slot`` makes per buffer -- asserted here from the spec side, since
        that is the query a caller holding specs actually has.
        """
        manager = self._quantised_manager(paired_scales=True)
        try:
            data = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            scales = IndexTableSpec([BufferId(0, "k_scale"), BufferId(0, "v_scale")])
            batch = self._batch(manager, [data, scales])

            table = batch.index_table_id(data)
            self.assertEqual(table, batch.index_table_id(scales), "mirrored specs are one table")
            # The spec's buffers sit at entries [0, n) in its own order, so the per-buffer
            # query adds nothing a holder of the spec did not already know.
            for entry, role in enumerate(("key", "value")):
                self.assertEqual(
                    _introspection.batch_index_table_slot(batch, 0, role), (table, entry)
                )
            for entry, role in enumerate(("k_scale", "v_scale")):
                self.assertEqual(
                    _introspection.batch_index_table_slot(batch, 0, role), (table, entry)
                )
        finally:
            manager.shutdown()

    def test_a_buffer_and_its_spec_reach_the_same_table_memory(self):
        """The two queries differ only in whether the caller still has to slice.

        A buffer's table is the one its spec names, so the pointer is the same and the
        entry is the whole of the difference -- which is what lets a caller hold specs and
        never ask about a buffer.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to read the view")
        # Unpaired, so the scale keeps a table of its own: with one table every buffer
        # would reach the right memory by having nowhere else to go.
        manager = self._quantised_manager(paired_scales=False)
        try:
            data = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            scale = IndexTableSpec([BufferId(0, "k_scale")])
            batch = self._batch(manager, [data, scale])
            self.assertEqual(batch.num_index_tables, 2, "the two specs must not have merged")

            for spec, roles in ((data, ("key", "value")), (scale, ("k_scale",))):
                owned = torch.from_dlpack(batch.converted_index_table(batch.index_table_id(spec)))
                other = next(s for s in (data, scale) if s is not spec)
                disowned = torch.from_dlpack(
                    batch.converted_index_table(batch.index_table_id(other))
                )
                self.assertNotEqual(owned.data_ptr(), disowned.data_ptr())
                for entry, role in enumerate(roles):
                    table, slot = _introspection.batch_index_table_slot(batch, 0, role)
                    from_buffer = torch.from_dlpack(batch.converted_index_table(table))
                    self.assertEqual(from_buffer.data_ptr(), owned.data_ptr(), role)
                    self.assertEqual(slot, entry, role)
                    # The entry addresses a plane of that same memory rather than another
                    # allocation, so a subset is a slice and never a table of its own.
                    self.assertEqual(
                        from_buffer[:, slot, :].data_ptr(),
                        owned.data_ptr() + slot * owned.stride(1) * owned.element_size(),
                        role,
                    )
        finally:
            manager.shutdown()

    def test_merged_specs_keep_pool_specs_of_their_own(self):
        """One table, two bases -- which is the whole mechanism of the merge.

        The data and its block scales share one set of converted entries and are told
        apart only by what they add those entries to, so the pool spec is a property of
        the spec and could not be answered for the table.
        """
        manager = self._quantised_manager(paired_scales=True)
        try:
            data = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            scales = IndexTableSpec([BufferId(0, "k_scale"), BufferId(0, "v_scale")])
            batch = self._batch(manager, [data, scales])
            self.assertEqual(batch.num_index_tables, 1, "the premise is that they merged")

            data_base, data_stride = batch.pool_spec_for(data)
            scale_base, scale_stride = batch.pool_spec_for(scales)
            self.assertNotEqual(data_base, scale_base, "one base would read one pool twice")
            # Separate pools of unequal buffer size, so the pages they count differ too.
            self.assertNotEqual(data_stride, scale_stride)
            for spec, base in ((data, data_base), (scales, scale_base)):
                for buffer in spec.buffers:
                    self.assertEqual(
                        _introspection.batch_index_table_base(batch, buffer.layer_id, buffer.role),
                        base,
                        "every buffer of a spec is reached through the spec's own base",
                    )
        finally:
            manager.shutdown()

    def test_a_pool_spec_needs_a_spec_the_batch_was_built_with(self):
        manager = self._quantised_manager(paired_scales=True)
        try:
            data = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            scales = IndexTableSpec([BufferId(0, "k_scale"), BufferId(0, "v_scale")])
            batch = self._batch(manager, [data, scales])
            with self.assertRaises(IndexError):
                batch.pool_spec_for(IndexTableSpec([BufferId(0, "key")]))
        finally:
            manager.shutdown()

    def test_a_spec_the_batch_was_not_built_with_has_no_table(self):
        manager = self._quantised_manager(paired_scales=True)
        try:
            data = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            scales = IndexTableSpec([BufferId(0, "k_scale"), BufferId(0, "v_scale")])
            batch = self._batch(manager, [data, scales])
            # A subset is not a spec: it names a slice of a table, not a table.
            with self.assertRaises(IndexError):
                batch.index_table_id(IndexTableSpec([BufferId(0, "key")]))
        finally:
            manager.shutdown()

    def test_specs_may_not_overlap(self):
        """A buffer belongs to one spec; a subset is read by slicing, not by declaring it.

        Both orders are refused, so this is a property of the pair rather than of
        whichever spec happened to be built first.
        """
        manager = self._quantised_manager(paired_scales=True)
        try:
            both = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
            lone = IndexTableSpec([BufferId(0, "key")])
            for order in ([both, lone], [lone, both]):
                with self.assertRaises(RuntimeError) as caught:
                    self._batch(manager, order)
                self.assertIn("may not overlap", str(caught.exception))
        finally:
            manager.shutdown()

    def test_a_spec_spanning_scales_is_refused(self):
        """A spec whose buffers disagree on scale cannot be served by one table.

        Scale sets the entry stride, so buffers with different scales have different
        converted indices for the same slot.
        """
        manager = self._quantised_manager(paired_scales=False)
        try:
            data_scale = manager.get_page_index_scale(0, "key")
            lone_scale = manager.get_page_index_scale(0, "k_scale")
            self.assertNotEqual(data_scale, lone_scale, "config does not produce differing scales")
            with self.assertRaises(Exception) as caught:
                self._batch(manager, [IndexTableSpec([BufferId(0, "key"), BufferId(0, "k_scale")])])
            self.assertIn("scales", str(caught.exception))
        finally:
            manager.shutdown()

    def test_a_spec_spanning_life_cycles_is_refused(self):
        """Different life cycles number their slots separately, so the entries cannot agree."""
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=32,
                cache_tiers=[GpuCacheTierConfig(quota=8 << 20)],
                layers=[
                    AttentionLayerConfig(
                        layer_id=LayerId(0),
                        buffers=[BufferConfig(role="key", size=1024)],
                        sliding_window_size=32,
                        num_sink_tokens=0,
                    ),
                    AttentionLayerConfig(
                        layer_id=LayerId(1),
                        buffers=[BufferConfig(role="key", size=1024)],
                        sliding_window_size=None,
                    ),
                ],
                constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=32, history_length=0)])],
            )
        )
        try:
            self.assertNotEqual(manager.get_layer_group_id(0), manager.get_layer_group_id(1))
            with self.assertRaises(Exception) as caught:
                self._batch(manager, [IndexTableSpec([BufferId(0, "key"), BufferId(1, "key")])])
            self.assertIn("life cycles", str(caught.exception))
        finally:
            manager.shutdown()

    def _assert_converted_matches_reference(
        self, manager, batch, cache, roles, mode, scratch=None, layer=0
    ):
        """Rebuild the concat table from PageIndexConverter and compare entry for entry.

        The reference gives each buffer its own table and puts the per-role offset in that
        table's base. A concat spec shares one base across its buffers, so it must carry
        the offset in the entry instead:

            expected[i] = reference[i] + (role_base[i] - spec_base) / stride

        Under SHARED that difference is the K-to-V gap; under PER_LAYER both bases are the
        pool group's, so it is zero and the reference's own layer_offset already separates
        the roles. Either way the reference supplies the arithmetic, so this tests the
        conversion rather than restating it.
        """
        lg = manager.get_layer_group_id(0)
        raw = [int(x) for x in cache.get_base_page_indices(lg)]
        self.assertTrue(any(x != BAD_PAGE_INDEX for x in raw), "no blocks to convert")

        converted = _introspection.batch_index_table(batch, 0)
        rows, blocks = batch.max_batch_size, batch.max_blocks_per_seq
        self.assertEqual(len(converted), len(roles) * rows * blocks)

        for entry, role in enumerate(roles):
            stride = manager.get_page_stride(layer, role)
            shift = (
                manager.get_mem_pool_base_address(layer, role, mode)
                - _introspection.batch_index_table_base(batch, layer, role)
            ) // stride
            reference = list(
                manager.get_page_index_converter(layer, role)(
                    raw[:blocks],
                    mode,
                    scratch.range if scratch else None,
                    scratch.slot_ids if scratch else [],
                )
            )
            want = [x if x == BAD_PAGE_INDEX else x + shift for x in reference]
            # Row-major in the entries: a row carries every entry before the next row
            # starts, which is what puts a spec's key and value planes side by side.
            base = (0 * len(roles) + entry) * blocks
            got = converted[base : base + blocks]
            self.assertEqual(got, want, f"entry {entry} ({role})")

    @contextlib.contextmanager
    def _published(self, manager, mode):
        """A one-member batch whose {key, value} spec has been converted once."""
        spec = IndexTableSpec([BufferId(0, "key"), BufferId(0, "value")])
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=4,
                    mode=mode,
                    index_tables=[spec],
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                tokens = 2 * 32
                # Seeded before joining: the Batch only accepts a history its row already
                # had the capacity for.
                self.assertTrue(cache.resize(tokens, tokens))
                batch.add_kv_cache(cache)
                batch.resize([tokens], [tokens])
                yield batch, cache
        finally:
            if cache is not None:
                if batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()

    def test_shared_entries_match_the_reference_converter(self):
        manager = self._manager(roles=("key", "value"), layers=1)
        try:
            with self._published(manager, PageIndexMode.SHARED) as (batch, cache):
                self._assert_converted_matches_reference(
                    manager, batch, cache, ("key", "value"), PageIndexMode.SHARED
                )
        finally:
            manager.shutdown()

    def test_per_layer_entries_match_the_reference_converter(self):
        """A separate case because the base differs between the modes.

        The pool group's base here, carrying no per-layer offset, against the spec's
        first buffer under SHARED, which carries one. Same formula, different entries.
        """
        manager = self._manager(roles=("key", "value"), layers=1)
        try:
            with self._published(manager, PageIndexMode.PER_LAYER) as (batch, cache):
                self._assert_converted_matches_reference(
                    manager, batch, cache, ("key", "value"), PageIndexMode.PER_LAYER
                )
        finally:
            manager.shutdown()

    def test_scratch_staging_covers_the_layer_taking_the_most_sub_pages(self):
        """A life cycle whose layers put unequal buffer counts in one pool.

        The slots a request takes come from the life cycle's widest sub-pages-per-slot
        ratio, so the staging has to be sized by that same ratio. Sizing it from whichever
        buffer an index table happens to name first under-counts whenever that buffer's
        layer is not the widest, and the request then holds more slots than its row has.
        """
        window, tpb, prompt_len = 128, 32, 1024
        # Layer 0 puts four buffers in the pool, layers 1-3 put two. Same size, so one
        # pool; same window, so one life cycle.
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[
                    BufferConfig(role=DataRole(role), size=8192)
                    for role in ("key", "value", "key2", "value2")
                ],
                sliding_window_size=window,
                num_sink_tokens=0,
            )
        ] + [
            AttentionLayerConfig(
                layer_id=LayerId(i),
                buffers=[
                    BufferConfig(role=DataRole("key"), size=8192),
                    BufferConfig(role=DataRole("value"), size=8192),
                ],
                sliding_window_size=window,
                num_sink_tokens=0,
            )
            for i in range(1, 4)
        ]
        cfg = KVCacheManagerConfig(
            tokens_per_block=tpb,
            cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
            layers=layers,
            swa_scratch_reuse=SwaScratchReuseConfig(max_rewind_len=0),
        )
        manager = KVCacheManager(cfg)
        cache = None
        batch = None
        try:
            lg = manager.get_layer_group_id(0)
            max_blocks = div_up(prompt_len, tpb)

            # The premise: the widest layer takes strictly more sub-pages per slot than a
            # two-buffer one, so the two sizings genuinely differ.
            widest = manager.get_page_index_converter(0, "key")
            narrow = manager.get_page_index_converter(1, "key")
            self.assertEqual(widest.scale, narrow.scale, "expected one pool for the group")
            self.assertGreater(
                widest.scratch_pages_per_block,
                narrow.scratch_pages_per_block,
                "config is uniform, so it cannot distinguish the two sizings",
            )

            # A table naming only a narrow layer's buffers must still stage the widest
            # layer's slots, because the request takes them for the whole life cycle.
            spec = IndexTableSpec([BufferId(1, "key"), BufferId(1, "value")])
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    index_tables=[spec],
                    max_batch_size=1,
                    max_blocks_per_seq=max_blocks,
                    mode=PageIndexMode.PER_LAYER,
                )
                cache = manager.create_kv_cache(None, [TokenId(i) for i in range(prompt_len)])
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                # Taking the slots is what overruns a row sized from the narrow buffer.
                self.assertTrue(cache.resize(prompt_len))
                self.assertTrue(cache.has_scratch_slots, "config produced no scratch")

                slots = cache.get_scratch_slots(lg)
                self.assertLessEqual(
                    len(slots),
                    manager.max_scratch_slots(max_blocks),
                    "the request holds more scratch slots than the manager sizes for",
                )
                batch.wait_published()
                batch.publish()
                batch.wait_published()
        finally:
            if batch is not None and cache is not None and batch.row_of(cache) is not None:
                batch.remove_kv_cache(cache)
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_scratch_blocks_match_the_reference_converter(self):
        """The substituting path, against PageIndexConverter given the same scratch range.

        A scratch block's slot comes from the spec rather than the page table, and the
        layer's offset wraps inside the shared slot instead of being added to it. Both are
        easy to transcribe wrongly, and neither shows up on a config without scratch, so
        this drives a real scratch range and defers the arithmetic to the reference.
        """
        num_layers, window, tpb, prompt_len = 4, 128, 32, 1024
        cfg = KVCacheManagerConfig(
            tokens_per_block=tpb,
            cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(i),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=8192),
                        BufferConfig(role=DataRole("value"), size=8192),
                    ],
                    sliding_window_size=window,
                    num_sink_tokens=0,
                )
                for i in range(num_layers)
            ],
            swa_scratch_reuse=SwaScratchReuseConfig(max_rewind_len=0),
        )
        manager = KVCacheManager(cfg)
        cache = None
        batch = None
        try:
            lg = manager.get_layer_group_id(0)
            # Scratch makes the layers stop sharing one table, so PER_LAYER is required.
            self.assertIsNone(manager.supports_index_mode(PageIndexMode.SHARED, lg))

            # The last layer, because its sub-page offsets are the largest: the wrap only
            # bites when total % scale + offset reaches scale, which layer zero's offsets
            # of {0, 1} never do.
            layer = num_layers - 1
            converter = manager.get_page_index_converter(layer, "value")
            scale, spb = converter.scale, converter.scratch_pages_per_block
            self.assertTrue(
                any((block * spb) % scale + converter.layer_offset >= scale for block in range(32)),
                "config never wraps the layer offset, so the scratch path is only half tested",
            )

            spec = IndexTableSpec([BufferId(layer, "key"), BufferId(layer, "value")])
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=div_up(prompt_len, tpb),
                    mode=PageIndexMode.PER_LAYER,
                    index_tables=[spec],
                )
                cache = manager.create_kv_cache(None, [TokenId(i) for i in range(prompt_len)])
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertTrue(cache.resize(prompt_len))
                self.assertTrue(cache.has_scratch_slots, "config produced no scratch")

                scratch = _scratch_desc(cache, lg)
                self.assertIsNotNone(scratch)
                self.assertGreater(scratch.range.end - scratch.range.beg, 0)

                batch.wait_published()
                batch.publish()
                self._assert_converted_matches_reference(
                    manager,
                    batch,
                    cache,
                    ("key", "value"),
                    PageIndexMode.PER_LAYER,
                    scratch=scratch,
                    layer=layer,
                )
        finally:
            if cache is not None:
                if batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_a_named_shared_serves_a_member_that_keeps_the_claim(self):
        """The case the escape hatch exists for: manager undecided, member clear.

        The manager answers for every request this sliding life cycle could carry, so it
        will not promise SHARED. A member whose capacity never reaches the window holds
        nothing stale, so it takes no scratch and says so itself -- and the batch that
        named SHARED publishes it against one table for all four layers.
        """
        num_layers, window, tpb = 4, 128, 32
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=tpb,
                cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
                layers=[
                    AttentionLayerConfig(
                        layer_id=LayerId(i),
                        buffers=[BufferConfig(role=DataRole("key"), size=8192)],
                        sliding_window_size=window,
                        num_sink_tokens=0,
                    )
                    for i in range(num_layers)
                ],
                swa_scratch_reuse=SwaScratchReuseConfig(max_rewind_len=0),
            )
        )
        cache = None
        batch = None
        try:
            lg = manager.get_layer_group_id(0)
            # The manager will not promise it...
            self.assertIsNone(manager.supports_index_mode(PageIndexMode.SHARED, lg))
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=4,
                    mode=PageIndexMode.SHARED,
                    index_tables=[IndexTableSpec([BufferId(0, "key")])],
                )
                # One table for the life cycle, not one per layer: the saving being claimed.
                self.assertEqual(batch.num_index_tables, 1)

                tokens = tpb  # One block, far short of the window, so nothing goes stale.
                cache = manager.create_kv_cache(None, [TokenId(i) for i in range(tokens)])
                self.assertTrue(cache.resume(handle))
                self.assertTrue(cache.resize(tokens))
                # ...but the member itself does, which is what makes the claim true.
                self.assertFalse(cache.has_scratch_slots)
                self.assertTrue(cache.supports_index_mode(PageIndexMode.SHARED, lg))

                batch.add_kv_cache(cache)
                batch.publish()
                batch.wait_published()
                # Published, and every layer still reads the one table.
                self.assertEqual(batch.index_mode(lg), PageIndexMode.SHARED)
                self.assertEqual(batch.num_index_tables, 1)
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_a_named_shared_is_held_to_its_claim_at_publish(self):
        """Naming SHARED asserts no member holds scratch; publish is what enforces it.

        Construction lets the claim stand because configuration cannot settle it -- the
        manager answers for every request the life cycle could carry. So the batch admits
        a member that does hold scratch, and the publish that would hand its layers one
        table for indices that differ between them refuses instead.
        """
        num_layers, window, tpb, prompt_len = 4, 128, 32, 1024
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=tpb,
                cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
                layers=[
                    AttentionLayerConfig(
                        layer_id=LayerId(i),
                        buffers=[BufferConfig(role=DataRole("key"), size=8192)],
                        sliding_window_size=window,
                        num_sink_tokens=0,
                    )
                    for i in range(num_layers)
                ],
                swa_scratch_reuse=SwaScratchReuseConfig(max_rewind_len=0),
            )
        )
        cache = None
        batch = None
        try:
            lg = manager.get_layer_group_id(0)
            self.assertIsNone(manager.supports_index_mode(PageIndexMode.SHARED, lg))
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                # Named, so construction allows what the manager would not derive.
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=div_up(prompt_len, tpb),
                    mode=PageIndexMode.SHARED,
                    index_tables=[IndexTableSpec([BufferId(0, "key")])],
                )
                self.assertEqual(batch.index_mode(lg), PageIndexMode.SHARED)

                # A prefill-sized grow is what takes scratch slots, so this member breaks
                # the claim the batch was built on.
                cache = manager.create_kv_cache(None, [TokenId(i) for i in range(prompt_len)])
                self.assertTrue(cache.resume(handle))
                self.assertTrue(cache.resize(prompt_len))
                self.assertTrue(cache.has_scratch_slots, "config produced no scratch")

                # Seating is outside the assertion: the claim is that the batch takes the
                # member and publish() is what refuses it, which a block covering both
                # would not tell apart.
                batch.add_kv_cache(cache)
                with self.assertRaises(RuntimeError) as caught:
                    batch.publish()
                self.assertIn("SHARED life cycle", str(caught.exception))
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_an_unrequested_buffer_has_no_slot(self):
        manager = self._manager(roles=("key", "value"))
        try:
            batch = self._batch(manager, [IndexTableSpec([BufferId(0, "key")])])
            with self.assertRaises(Exception) as caught:
                _introspection.batch_index_table_slot(batch, 0, "value")
            self.assertIn("no index table spec", str(caught.exception))
        finally:
            manager.shutdown()

    def test_no_specs_means_no_tables(self):
        manager = self._manager(roles=("key",))
        try:
            self.assertEqual(self._batch(manager, []).num_index_tables, 0)
        finally:
            manager.shutdown()


class TestIndexMode(unittest.TestCase):
    """Which page index mode a life cycle gets, and why it is a per-life-cycle question.

    Only scratch-slot substitution can make one layer's converted indices differ from
    another's in the same life cycle. A life cycle with no sliding window yields an empty
    scratch range whatever the watermarks, so no request can hold scratch slots there and
    SHARED is always available -- even with scratch reuse enabled for the manager.
    """

    #: Layer 0 slides and so may hold scratch; layer 1 does not and so never can.
    WINDOWED_LAYER = 0
    PLAIN_LAYER = 1

    def setUp(self) -> None:
        init_cuda_once()

    @staticmethod
    def _manager(*, scratch_reuse: bool) -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=32,
                cache_tiers=[GpuCacheTierConfig(quota=4 << 20)],
                layers=[
                    AttentionLayerConfig(
                        layer_id=LayerId(0),
                        buffers=[BufferConfig(role=Role.KEY, size=1024)],
                        sliding_window_size=32,
                        num_sink_tokens=0,
                    ),
                    AttentionLayerConfig(
                        layer_id=LayerId(1),
                        buffers=[BufferConfig(role=Role.KEY, size=1024)],
                        sliding_window_size=None,
                    ),
                ],
                constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=32, history_length=0)])],
                swa_scratch_reuse=(SwaScratchReuseConfig() if scratch_reuse else None),
            )
        )

    def test_a_life_cycle_without_a_window_keeps_shared_under_scratch_reuse(self):
        """The collapsing query cannot say this: it answers for the manager, not a group."""
        manager = self._manager(scratch_reuse=True)
        try:
            windowed = manager.get_layer_group_id(self.WINDOWED_LAYER)
            plain = manager.get_layer_group_id(self.PLAIN_LAYER)
            self.assertNotEqual(windowed, plain)

            # Per life cycle: only the sliding one is in doubt.
            self.assertIsNone(manager.supports_index_mode(PageIndexMode.SHARED, windowed))
            self.assertTrue(manager.supports_index_mode(PageIndexMode.SHARED, plain))
            # PER_LAYER never depends on anything.
            self.assertTrue(manager.supports_index_mode(PageIndexMode.PER_LAYER, windowed))
        finally:
            manager.shutdown()

    def test_without_scratch_reuse_every_life_cycle_is_shared(self):
        manager = self._manager(scratch_reuse=False)
        try:
            for layer in (self.WINDOWED_LAYER, self.PLAIN_LAYER):
                lg = manager.get_layer_group_id(layer)
                self.assertTrue(manager.supports_index_mode(PageIndexMode.SHARED, lg))
        finally:
            manager.shutdown()

    def test_a_batch_takes_the_smaller_table_wherever_it_can(self):
        """Per life cycle, not per batch: the sliding group alone pays for PER_LAYER."""
        manager = self._manager(scratch_reuse=True)
        try:
            batch = Batch(
                manager, NULL_STREAM, index_tables=[], max_batch_size=1, max_blocks_per_seq=4
            )
            self.assertEqual(
                batch.index_mode(manager.get_layer_group_id(self.WINDOWED_LAYER)),
                PageIndexMode.PER_LAYER,
            )
            self.assertEqual(
                batch.index_mode(manager.get_layer_group_id(self.PLAIN_LAYER)),
                PageIndexMode.SHARED,
            )
        finally:
            manager.shutdown()

    def test_an_explicit_mode_overrides_every_life_cycle(self):
        manager = self._manager(scratch_reuse=False)
        try:
            batch = Batch(
                manager,
                NULL_STREAM,
                index_tables=[],
                max_batch_size=1,
                max_blocks_per_seq=4,
                mode=PageIndexMode.PER_LAYER,
            )
            for layer in (self.WINDOWED_LAYER, self.PLAIN_LAYER):
                lg = manager.get_layer_group_id(layer)
                # SHARED was available and still not taken.
                self.assertTrue(manager.supports_index_mode(PageIndexMode.SHARED, lg))
                self.assertEqual(batch.index_mode(lg), PageIndexMode.PER_LAYER)
        finally:
            manager.shutdown()

    def test_naming_shared_takes_it_where_the_manager_will_not_promise_it(self):
        """A named SHARED is a claim about this batch's members, not about requests.

        The manager answers for every request a life cycle could carry, so a sliding one
        stays undecided. A batch that will only ever hold decode members knows better:
        scratch covers the blocks a prefill chunk adds, and a one-token request adds none.
        """
        manager = self._manager(scratch_reuse=True)
        try:
            windowed = manager.get_layer_group_id(self.WINDOWED_LAYER)
            self.assertIsNone(manager.supports_index_mode(PageIndexMode.SHARED, windowed))

            spec = IndexTableSpec([BufferId(self.WINDOWED_LAYER, Role.KEY)])
            batch = Batch(
                manager,
                NULL_STREAM,
                max_batch_size=1,
                max_blocks_per_seq=4,
                mode=PageIndexMode.SHARED,
                index_tables=[spec],
            )
            self.assertEqual(batch.index_mode(windowed), PageIndexMode.SHARED)
            # One table for the life cycle, which is the whole point of taking SHARED.
            self.assertEqual(batch.num_index_tables, 1)
        finally:
            manager.shutdown()

    def test_a_derived_shared_is_still_refused_where_scratch_is_possible(self):
        """Only a named mode may outrun the manager: nothing would check a derived one."""
        manager = self._manager(scratch_reuse=True)
        try:
            batch = Batch(
                manager,
                NULL_STREAM,
                max_batch_size=1,
                max_blocks_per_seq=4,
                index_tables=[IndexTableSpec([BufferId(self.WINDOWED_LAYER, Role.KEY)])],
            )
            # Derived, so the sliding life cycle took the table it can prove correct.
            self.assertEqual(
                batch.index_mode(manager.get_layer_group_id(self.WINDOWED_LAYER)),
                PageIndexMode.PER_LAYER,
            )
        finally:
            manager.shutdown()


class TestBatchPageTable(unittest.TestCase):
    """The Batch's own host page-index staging: who writes it, and what publish does with it.

    A member writes its indices into the Batch's row and nowhere else, which is what keeps
    one buffer with one writer and a binding that survives the whole decode. A consumer
    needing the same numbers reads coalesced_page_table() rather than keeping a buffer of its
    own.
    """

    ROWS = 2
    BLOCKS = 4
    GROUPS = 1

    def setUp(self) -> None:
        init_cuda_once()

    def _torch(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to read the staging as an array")
        return torch

    @contextlib.contextmanager
    def _fixture(self):
        """A manager and a Batch staging its own page indices."""
        torch = self._torch()
        manager = TestSparseFetch._manager(("key",))
        try:
            # The stream belongs to the fixture because the Batch takes it at construction:
            # a member joins the stream it was built for, so one cannot outlive the other.
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=self.ROWS,
                    max_blocks_per_seq=self.BLOCKS,
                    index_tables=manager.sparse_index_table_specs(),
                )
                yield torch, manager, batch, handle
        finally:
            manager.shutdown()

    def _row(self, torch, batch, manager, row: int, group: int = 0):
        lg = manager.get_layer_group_id(group)
        host = torch.from_dlpack(batch.coalesced_page_table(lg))
        return host[row].tolist()

    @contextlib.contextmanager
    def _member(self, manager, batch, row: int, handle):
        """A resumed cache seated in `row`, with two whole blocks of history."""
        cache = manager.create_kv_cache()
        try:
            self.assertTrue(cache.resume(handle))
            tokens = 2 * TestSparseFetch.TOKENS_PER_BLOCK
            # Seeded before joining: the Batch only accepts a history its row already had
            # the capacity for.
            self.assertTrue(cache.resize(tokens, tokens))
            batch.add_kv_cache(cache, row)
            sizes = [None] * self.ROWS
            sizes[row] = tokens
            batch.resize(list(sizes), list(sizes))
            yield cache
        finally:
            if batch.row_of(cache) is not None:
                batch.remove_kv_cache(cache)
            cache.close()

    def test_a_member_writes_its_indices_into_its_own_row(self):
        """The row it was seated in, and no other."""
        with self._fixture() as (torch, manager, batch, handle):
            lg = manager.get_layer_group_id(0)
            with self._member(manager, batch, 1, handle) as cache:
                expected = [int(x) for x in cache.get_base_page_indices(lg)][: self.BLOCKS]
                self.assertEqual(self._row(torch, batch, manager, 1), expected)
                # A page index, not the fill: the member really has pages here.
                self.assertNotEqual(expected[0], BAD_PAGE_INDEX)
                # The row nobody occupies still reads as absent.
                self.assertEqual(
                    self._row(torch, batch, manager, 0), [BAD_PAGE_INDEX] * self.BLOCKS
                )

    def test_the_host_view_is_what_publish_uploads(self):
        """The device table is a snapshot of the host rows, row for row.

        It is copied rather than read in place because publish() is a snapshot: the host
        has to be free to write the next step's rows long before the kernels reading this
        step's have retired.
        """
        with self._fixture() as (torch, manager, batch, handle):
            lg = manager.get_layer_group_id(0)
            with self._member(manager, batch, 1, handle):
                # _member's resize() already published; republishing needs the wait.
                batch.wait_published()
                batch.publish()
                torch.cuda.synchronize()
                table = _introspection.batch_page_table(batch, lg)
                self.assertEqual(len(table), self.ROWS * self.BLOCKS)
                expected = self._row(torch, batch, manager, 1)
                self.assertEqual(table[self.BLOCKS : 2 * self.BLOCKS], expected)
                self.assertNotEqual(expected[0], BAD_PAGE_INDEX)
                self.assertEqual(table[: self.BLOCKS], [BAD_PAGE_INDEX] * self.BLOCKS)

    def test_publishing_over_a_publish_in_flight_is_refused(self):
        """A wait here would be too late, so it is the caller's to make.

        Everything that writes the staged rows has to be separated from the copy reading
        them: the members' own mutations, and the watermarks and views publish() stages
        itself. Waiting inside publish() would cover only the second, and would do it while
        hiding that a member mutated before the call had already raced.
        """
        with self._fixture() as (torch, manager, batch, handle):
            with self._member(manager, batch, 0, handle):
                # _member's resize() published and nobody has waited on it.
                with self.assertRaises(Exception) as caught:
                    batch.publish()
                self.assertIn("still reading the staged rows", str(caught.exception))

                batch.wait_published()
                batch.publish()

    def test_a_member_refuses_a_second_page_index_buffer(self):
        """Its indices live in the Batch's row, so a second buffer is refused.

        That buffer would take the writes with it and strand the row at whatever it held
        when the cache joined -- silently, which is the one outcome worth a hard failure.
        """
        with self._fixture() as (torch, manager, batch, handle):
            lg = manager.get_layer_group_id(0)
            with self._member(manager, batch, 0, handle) as cache:
                self.assertEqual(cache.batch_row, 0)
                spare = array.array("i", [BAD_PAGE_INDEX] * self.BLOCKS)
                with self.assertRaises(Exception) as caught:
                    cache.set_base_page_index_buf(DEFAULT_BEAM_INDEX, lg, memoryview(spare))
                self.assertIn("Batch member", str(caught.exception))

    def test_a_seated_member_cannot_suspend(self):
        """Unlocking pages writes through the binding, which is the Batch's row.

        Doing that while seated would write a row a publish may still be reading, so the
        member leaves first and the row is cleared by the removal.
        """
        with self._fixture() as (torch, manager, batch, handle):
            with self._member(manager, batch, 1, handle) as cache:
                with self.assertRaisesRegex(Exception, "remove it from the Batch first"):
                    cache.suspend()
                batch.remove_kv_cache(cache)
                self.assertEqual(cache.batch_row, None)
                self.assertEqual(
                    self._row(torch, batch, manager, 1), [BAD_PAGE_INDEX] * self.BLOCKS
                )
                cache.suspend()


class TestSelectionId(unittest.TestCase):
    """What a selection_id may name, which decides when two layers share a selection."""

    TOKENS_PER_BLOCK = 128

    @classmethod
    def _config(cls, ids: list[int | None], **overrides) -> KVCacheManagerConfig:
        """One sparse layer per entry of `ids`, each naming that selection or none."""

        def cache(layer: int, selection_id: int | None) -> SparseCacheConfig:
            return SparseCacheConfig(
                num_selected_blocks=overrides.get("num_selected_blocks", {}).get(layer, 2),
                max_cached_units=overrides.get("max_cached_units", {}).get(layer, 4),
                unit_tokens=overrides.get("unit_tokens", {}).get(layer, cls.TOKENS_PER_BLOCK),
                selection_id=selection_id,
            )

        return KVCacheManagerConfig(
            tokens_per_block=cls.TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(64 << 20)],
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 2)],
            layers=[
                AttentionLayerConfig(
                    LayerId(layer),
                    [
                        BufferConfig("key", 4096, is_sparse=True),
                        BufferConfig("index", 256),
                    ],
                    sparse_cache=cache(layer, selection_id),
                )
                for layer, selection_id in enumerate(ids)
            ],
        )

    def test_an_unset_id_defaults_to_the_layers_own(self):
        self.assertIsNone(
            SparseCacheConfig(
                num_selected_blocks=2, max_cached_units=4, unit_tokens=self.TOKENS_PER_BLOCK
            ).selection_id
        )
        named = SparseCacheConfig(
            num_selected_blocks=2,
            max_cached_units=4,
            unit_tokens=self.TOKENS_PER_BLOCK,
            selection_id=7,
        )
        self.assertEqual(named.selection_id, 7)
        self.assertEqual(deepcopy(named).selection_id, 7)

    def test_layers_may_share_one_selection(self):
        self._config([3, 3, None])

    def test_silence_is_never_shared(self):
        """Layer 1 names nothing, so it reads as selection 1; layer 0 may not take it."""
        with self.assertRaises(AssertionError) as caught:
            self._config([1, None])
        self.assertIn("names a layer that set none of its own", str(caught.exception))

    def test_sharers_must_agree_on_what_they_select(self):
        with self.assertRaises(AssertionError) as caught:
            self._config([5, 5], max_cached_units={1: 8})
        self.assertIn("must agree on", str(caught.exception))

    def test_a_layer_may_name_its_own_id(self):
        """Naming yourself is not sharing, so it collides with nothing."""
        self._config([0, None])


class TestSparseSelectionSharing(unittest.TestCase):
    """What two layers given one selection resolve to.

    Which decides whether a shared selection may share one resident set.

    A layer's resolve maintains that layer's own residency and recency. When several
    layers of a life cycle are handed the same selection -- which is what a model with
    one indexer across several layers does -- the question is whether those per-layer
    answers differ. They do not, and this pins that: same starting state, same input,
    same evolution. It is the premise a shared selection group rests on, so it is
    checked here rather than assumed.
    """

    TOKENS_PER_BLOCK = 128
    SELECTED = 2
    CACHED_UNITS = 4
    SPARSE_BYTES = 4096
    LAYERS = 2

    def setUp(self) -> None:
        init_cuda_once()

    @classmethod
    def _manager(cls, selection_id: int | None = None) -> KVCacheManager:
        """Two sparse layers alike in every way, so one life cycle holds both.

        With `selection_id` they also name one selection, which is the whole
        difference between resolving twice and resolving once.
        """
        sparse_cache = SparseCacheConfig(
            num_selected_blocks=cls.SELECTED,
            max_cached_units=cls.CACHED_UNITS,
            unit_tokens=cls.TOKENS_PER_BLOCK,
            selection_id=selection_id,
        )
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=cls.TOKENS_PER_BLOCK,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 4)
                ],
                layers=[
                    AttentionLayerConfig(
                        LayerId(layer),
                        [
                            BufferConfig("key", cls.SPARSE_BYTES, is_sparse=True),
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=sparse_cache,
                    )
                    for layer in range(cls.LAYERS)
                ],
            )
        )

    def test_one_selection_resolves_alike_in_every_layer(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to pass device arrays")

        manager = self._manager()
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                lc_id = manager.get_layer_group_id(LayerId(0))
                self.assertEqual(lc_id, manager.get_layer_group_id(LayerId(1)))

                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=2,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                tokens = 2 * self.TOKENS_PER_BLOCK
                self.assertTrue(cache.resize(tokens, tokens))
                batch.add_kv_cache(cache)
                self.assertEqual(batch.resize([tokens, None], [tokens, None]), [True, False])

                selected = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")
                for layer in range(self.LAYERS):
                    out = torch.full((2, 1, self.SELECTED), 12345, dtype=torch.int32, device="cuda")
                    batch.fetch_sparse_pages([BufferId(layer, "key")], selected, out, handle)
                torch.cuda.synchronize()

                ordinals = _introspection.sparse_cache_resident_ordinals(cache, lc_id)
                entries = len(ordinals) // self.LAYERS
                self.assertEqual(len(ordinals), entries * self.LAYERS)
                first = ordinals[:entries]
                # Something was admitted, or the comparison below is vacuous.
                self.assertNotEqual(first, [BAD_PAGE_INDEX] * entries, "no unit became resident")
                for layer in range(1, self.LAYERS):
                    self.assertEqual(
                        ordinals[layer * entries : (layer + 1) * entries],
                        first,
                        f"layer {layer} resolved the same selection differently",
                    )
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            del batch
            manager.shutdown()

    def test_layers_naming_one_selection_keep_one_resident_set(self):
        """One resolve, one metadata row, and each layer's own mirror of its entries."""
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to pass device arrays")

        manager = self._manager(selection_id=0)
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                lc_id = manager.get_layer_group_id(LayerId(0))

                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=2,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                tokens = 2 * self.TOKENS_PER_BLOCK
                self.assertTrue(cache.resize(tokens, tokens))
                batch.add_kv_cache(cache)
                self.assertEqual(batch.resize([tokens, None], [tokens, None]), [True, False])

                layers = [[BufferId(layer, "key")] for layer in range(self.LAYERS)]
                selected = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32, device="cuda")

                # The fused call cannot serve a shared selection, and says so rather
                # than resolving it a second time.
                with self.assertRaises(Exception) as caught:
                    out = torch.empty((2, 1, self.SELECTED), dtype=torch.int32, device="cuda")
                    batch.fetch_sparse_pages(layers[0], selected, out, handle)
                self.assertIn("shares its selection_id", str(caught.exception))

                # Resolved once for both layers, then copied a layer at a time.
                batch.resolve_sparse_selection(layers[0], selected, handle)
                outs = []
                for buffers in layers:
                    out = torch.full((2, 1, self.SELECTED), 12345, dtype=torch.int32, device="cuda")
                    batch.fetch_resolved_pages(buffers, selected, out, handle)
                    outs.append(out)
                torch.cuda.synchronize()

                # One metadata row for the two layers, where selecting apart gave two.
                entries = _introspection.batch_sparse_entries_per_row(batch, lc_id)[0]
                ordinals = _introspection.sparse_cache_resident_ordinals(cache, lc_id)
                self.assertEqual(len(ordinals), entries)
                self.assertNotEqual(ordinals, [BAD_PAGE_INDEX] * entries, "no unit became resident")

                # The entry each unit got is the selection's, so the two layers differ
                # only by where their cells sit: one buffer's worth of them apart.
                cells_per_buffer = entries  # unit_tokens == tokens_per_block
                first = outs[0].tolist()
                for layer in range(1, self.LAYERS):
                    shifted = [
                        [
                            [
                                index - layer * cells_per_buffer
                                if index != BAD_PAGE_INDEX
                                else index
                                for index in slots
                            ]
                            for slots in row
                        ]
                        for row in outs[layer].tolist()
                    ]
                    self.assertEqual(shifted, first, f"layer {layer} landed on a different entry")
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            del batch
            manager.shutdown()


class TestSparseDataPath(unittest.TestCase):
    """The bytes themselves, end to end, against real hot pages.

    Every other sparse test asserts residency, index shape or counts. Those pass whatever
    base address the copy uses, so they cannot see a page side pointed at the wrong
    allocation -- which is exactly what moving history to the hot tier changed. These two
    read and write the actual memory on both sides of the copy.
    """

    TPB = 128
    BUF = 4096
    SELECTED = 2
    CELLS = 4

    def setUp(self) -> None:
        init_cuda_once()

    @classmethod
    def _manager(cls) -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=cls.TPB,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 2)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                "key",
                                cls.BUF,
                                is_sparse=True,
                            ),
                            # A pool group is sized from its device-backed pools, so a sparse layer
                            # carries one. This is also the shape a model uses: an indexer and the
                            # data it selects over are one attention computation.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=cls.SELECTED,
                            max_cached_units=cls.CELLS,
                            unit_tokens=cls.TPB,
                        ),
                    )
                ],
            )
        )

    def _deps(self):
        try:
            import ctypes

            import cuda.bindings.runtime as cudart
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch and the CUDA bindings are needed to address raw memory")
        return ctypes, cudart, torch

    @staticmethod
    def _pattern(size: int, seed: int) -> bytes:
        return bytes((i * seed + 13) % 251 for i in range(size))

    def test_a_fetched_cell_holds_the_bytes_of_its_hot_page(self):
        ctypes, cudart, torch = self._deps()
        manager = self._manager()
        cache = None
        try:
            lg = manager.get_layer_group_id(0)
            pool = int(manager.get_mem_pool_base_address(LayerId(0), "key"))
            stride = manager.get_page_stride(LayerId(0), "key")
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                arena = int(batch.get_sparse_cache_base_address(LayerId(0), "key"))
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertTrue(cache.resize(4 * self.TPB, 4 * self.TPB))

                # Paint one block's hot page. It is host memory, so the host writes it
                # directly; the kernel reaches the same bytes through the device mapping.
                ordinal = 1
                pattern = self._pattern(self.BUF, 7)
                pages = [int(x) for x in cache.get_base_page_indices(lg)]
                ctypes.memmove(pool + pages[ordinal] * stride, pattern, self.BUF)

                batch.publish()
                selected = torch.tensor([[ordinal, -1]], dtype=torch.int32, device="cuda")
                out = torch.full((1, 1, self.SELECTED), 12345, dtype=torch.int32, device="cuda")
                batch.fetch_sparse_pages([BufferId(0, "key")], selected, out, handle)
                torch.cuda.synchronize()

                index = int(out[0][0][0])
                self.assertNotEqual(index, BAD_PAGE_INDEX)
                got = bytearray(self.BUF)
                sink = (ctypes.c_ubyte * self.BUF).from_buffer(got)
                (err,) = cudart.cudaMemcpy(
                    ctypes.addressof(sink),
                    arena + index * self.BUF,
                    self.BUF,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
                self.assertEqual(int(err), 0)
                self.assertEqual(bytes(got), pattern, "the cell does not hold its page's bytes")
                # Removal returns the lease and discards the GPU cache, so the
                # write-back record has to match the history being given up.
                batch.flush_sparse_units([4 * self.TPB])
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_a_flushed_cell_lands_in_its_hot_page(self):
        ctypes, cudart, torch = self._deps()
        manager = self._manager()
        cache = None
        try:
            lg = manager.get_layer_group_id(0)
            pool = int(manager.get_mem_pool_base_address(LayerId(0), "key"))
            stride = manager.get_page_stride(LayerId(0), "key")
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                arena = int(batch.get_sparse_cache_base_address(LayerId(0), "key"))
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                # Capacity one token past a watermark that has not reached unit 0, so unit 0
                # is in flight and the resolve pins a cell for it.
                self.assertTrue(cache.resize(self.TPB, self.TPB - 1))
                batch.publish()
                selected = torch.tensor([[-1, -1]], dtype=torch.int32, device="cuda")
                out = torch.full((1, 1, self.SELECTED), 12345, dtype=torch.int32, device="cuda")
                batch.fetch_sparse_pages([BufferId(0, "key")], selected, out, handle)
                torch.cuda.synchronize()

                resident = _introspection.sparse_cache_resident_ordinals(cache, lg)
                self.assertIn(0, resident, "the in-flight unit must hold a cell")
                cell = _introspection.sparse_cache_base_cell(cache, lg) + resident.index(0)

                pattern = self._pattern(self.BUF, 11)
                source = (ctypes.c_ubyte * self.BUF).from_buffer(bytearray(pattern))
                (err,) = cudart.cudaMemcpy(
                    arena + cell * self.BUF,
                    ctypes.addressof(source),
                    self.BUF,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
                self.assertEqual(int(err), 0)

                batch.wait_published()
                # The watermark crosses unit 0, so its cell is written back. The flush runs
                # before the move, since a block must reach its host page before it becomes
                # selectable.
                batch.flush_sparse_units([self.TPB])
                self.assertTrue(cache.resize(self.TPB + 1, self.TPB))
                torch.cuda.synchronize()

                pages = [int(x) for x in cache.get_base_page_indices(lg)]
                landed = ctypes.string_at(pool + pages[0] * stride, self.BUF)
                self.assertEqual(landed, pattern, "the write-back did not reach the hot page")
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()


class TestSparseWriteIndices(unittest.TestCase):
    """prepare_sparse_write_indices: the cell the model writes its new unit into.

    The unit between the two watermarks has no host page yet, so nothing can be fetched for
    it; what the model needs is only where to put it. These cover that the reported cell is
    the one the write-back later reads, that asking twice does not move it, and that a fetch
    running afterwards leaves it alone.
    """

    TPB = 128
    BUF = 4096
    SELECTED = 2
    CELLS = 4

    def setUp(self) -> None:
        init_cuda_once()

    @classmethod
    def _manager(cls) -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=cls.TPB,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 2)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                "key",
                                cls.BUF,
                                is_sparse=True,
                            ),
                            # A pool group is sized from its device-backed pools, so a sparse layer
                            # carries one. This is also the shape a model uses: an indexer and the
                            # data it selects over are one attention computation.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=cls.SELECTED,
                            max_cached_units=cls.CELLS,
                            unit_tokens=cls.TPB,
                        ),
                    )
                ],
            )
        )

    def _deps(self):
        try:
            import ctypes

            import cuda.bindings.runtime as cudart
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch and the CUDA bindings are needed to address raw memory")
        return ctypes, cudart, torch

    @staticmethod
    def _pattern(size: int, seed: int) -> bytes:
        return bytes((i * seed + 13) % 251 for i in range(size))

    def test_the_prepared_cell_is_the_one_written_back(self):
        """The whole contract in one path: write where it says, and the page gets it."""
        ctypes, cudart, torch = self._deps()
        manager = self._manager()
        cache = None
        try:
            lg = manager.get_layer_group_id(0)
            pool = int(manager.get_mem_pool_base_address(LayerId(0), "key"))
            stride = manager.get_page_stride(LayerId(0), "key")
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                arena = int(batch.get_sparse_cache_base_address(LayerId(0), "key"))
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                # Capacity one token past a history that has not reached unit 0, so unit 0
                # is the unit in flight.
                self.assertTrue(cache.resize(self.TPB, self.TPB - 1))
                batch.publish()

                out = torch.full((1, 1), 12345, dtype=torch.int32, device="cuda")
                batch.prepare_sparse_write_indices([BufferId(0, "key")], out, handle)
                torch.cuda.synchronize()
                index = int(out[0][0])
                self.assertNotEqual(index, BAD_PAGE_INDEX, "a unit is in flight")

                pattern = self._pattern(self.BUF, 41)
                source = (ctypes.c_ubyte * self.BUF).from_buffer(bytearray(pattern))
                (err,) = cudart.cudaMemcpy(
                    arena + index * self.BUF,
                    ctypes.addressof(source),
                    self.BUF,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
                self.assertEqual(int(err), 0)

                batch.wait_published()
                # The watermark crosses unit 0, so its cell is written back. The flush runs
                # before the move, since a block must reach its host page before it becomes
                # selectable.
                batch.flush_sparse_units([self.TPB])
                self.assertTrue(cache.resize(self.TPB + 1, self.TPB))
                torch.cuda.synchronize()

                pages = [int(x) for x in cache.get_base_page_indices(lg)]
                landed = ctypes.string_at(pool + pages[0] * stride, self.BUF)
                self.assertEqual(landed, pattern, "the prepared cell was not the one written back")
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_preparing_twice_reports_the_same_cell(self):
        """Idempotent while the watermarks hold: the second call must not move the unit."""
        _, _, torch = self._deps()
        manager = self._manager()
        cache = None
        try:
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertTrue(cache.resize(self.TPB, self.TPB - 1))
                batch.publish()

                seen = []
                for _ in range(3):
                    out = torch.full((1, 1), 12345, dtype=torch.int32, device="cuda")
                    batch.prepare_sparse_write_indices([BufferId(0, "key")], out, handle)
                    torch.cuda.synchronize()
                    seen.append(int(out[0][0]))
                self.assertNotEqual(seen[0], BAD_PAGE_INDEX)
                self.assertEqual(len(set(seen)), 1, f"the cell moved between calls: {seen}")
                # Removal returns the lease and discards the GPU cache, so the
                # write-back record has to match the history being given up.
                batch.flush_sparse_units([self.TPB - 1])
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_a_row_with_nothing_in_flight_reports_bad_page_index(self):
        """Both watermarks on a unit boundary: there is no unit to write."""
        _, _, torch = self._deps()
        manager = self._manager()
        cache = None
        try:
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=2,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertTrue(cache.resize(self.TPB, self.TPB))
                batch.publish()

                out = torch.full((2, 1), 12345, dtype=torch.int32, device="cuda")
                batch.prepare_sparse_write_indices([BufferId(0, "key")], out, handle)
                torch.cuda.synchronize()
                self.assertEqual(
                    int(out[0][0]), BAD_PAGE_INDEX, "capacity holds no unit above history"
                )
                # Written in full: the unoccupied row must be overwritten, not left at 12345.
                self.assertEqual(
                    int(out[1][0]), BAD_PAGE_INDEX, "a free row carries no reservation"
                )
                # Removal returns the lease and discards the GPU cache, so the
                # write-back record has to match the history being given up.
                batch.flush_sparse_units([self.TPB, 0])
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_every_buffer_of_a_layer_gets_its_own_cell(self):
        """Two buffers, and a headroom wider than what is actually in flight.

        The resolve's in-flight array is a row per request of max_cached_units minus the
        selection width, while at most one unit is ever in flight. Reading it with the wrong
        row stride writes one buffer's slots over the next buffer's output, which this sees
        as the second buffer losing its index.
        """
        _, _, torch = self._deps()
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=self.TPB,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 2)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                role,
                                self.BUF,
                                is_sparse=True,
                                # Headroom of CELLS - SELECTED = 2, wider than the one unit
                                # that can be in flight.
                            )
                            for role in ("key", "value")
                        ]
                        + [
                            # A pool group sizes its slots from its device-backed pools.
                            BufferConfig("index", 256)
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=self.SELECTED,
                            max_cached_units=self.CELLS,
                            unit_tokens=self.TPB,
                        ),
                    )
                ],
            )
        )
        cache = None
        try:
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertTrue(cache.resize(self.TPB, self.TPB - 1))
                batch.publish()

                out = torch.full((1, 2), 12345, dtype=torch.int32, device="cuda")
                batch.prepare_sparse_write_indices(
                    [BufferId(0, "key"), BufferId(0, "value")], out, handle
                )
                torch.cuda.synchronize()
                got = [int(out[0][0]), int(out[0][1])]
                self.assertNotIn(
                    BAD_PAGE_INDEX, got, f"both buffers hold the in-flight unit: {got}"
                )
                self.assertEqual(len(set(got)), 2, f"buffers must not share a cell: {got}")
                # Removal returns the lease and discards the GPU cache, so the
                # write-back record has to match the history being given up.
                batch.flush_sparse_units([self.TPB - 1])
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_each_layer_reports_a_cell_of_its_own(self):
        """One unit ordinal, one index per layer -- never one answer for the step.

        A layer resolves against its own resident set and its own recency order, so the same
        ordinal lands on a different entry in each; and each layer's buffer owns its own run
        of cells, so even equal entries would not collide. Both are why the call takes one
        layer rather than a group.
        """
        _, _, torch = self._deps()
        layers = 3
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=self.TPB,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 2)
                ],
                layers=[
                    AttentionLayerConfig(
                        layer,
                        [
                            BufferConfig(
                                "key",
                                self.BUF,
                                is_sparse=True,
                            ),
                            # A pool group is sized from its device-backed pools, so a sparse layer
                            # carries one. This is also the shape a model uses: an indexer and the
                            # data it selects over are one attention computation.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=self.SELECTED,
                            max_cached_units=self.CELLS,
                            unit_tokens=self.TPB,
                        ),
                    )
                    for layer in range(layers)
                ],
            )
        )
        cache = None
        try:
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertTrue(cache.resize(self.TPB, self.TPB - 1))
                batch.publish()

                indices = []
                for layer in range(layers):
                    out = torch.full((1, 1), 12345, dtype=torch.int32, device="cuda")
                    batch.prepare_sparse_write_indices(
                        [BufferId(LayerId(layer), "key")], out, handle
                    )
                    torch.cuda.synchronize()
                    indices.append(int(out[0][0]))

                self.assertNotIn(BAD_PAGE_INDEX, indices, "every layer has the same unit in flight")
                self.assertEqual(
                    len(set(indices)), layers, f"layers must not share a cell: {indices}"
                )
                # Each layer's buffer owns an adjacent run of cells_per_buffer cells, so the
                # runs the three indices fall in are consecutive.
                runs = sorted(index // self.CELLS for index in indices)
                self.assertEqual(
                    runs,
                    list(range(runs[0], runs[0] + layers)),
                    f"cell runs are not per buffer: {indices}",
                )
                # Removal returns the lease and discards the GPU cache, so the
                # write-back record has to match the history being given up.
                batch.flush_sparse_units([self.TPB - 1])
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_a_fetch_neither_moves_nor_overwrites_the_prepared_cell(self):
        """A fetch runs its own resolve; the prepared cell must come through unchanged.

        It covers the copy list: an in-flight unit's host page holds nothing yet, so a fetch
        that treated its entry as an ordinary miss would copy stale bytes over what the model
        wrote. Removing that guard fails this.

        It does not cover the separate `pinned` term in the victim test. Every in-flight
        ordinal is probed in the resolve's required prefix, so its entry is always classified
        hit or fetched and never `kClassUntouched` -- which the victim test already excludes.
        Dropping `pinned` leaves every test here passing, so it appears to be redundant with
        that classification rather than load-bearing.
        """
        ctypes, cudart, torch = self._deps()
        manager = self._manager()
        cache = None
        try:
            lg = manager.get_layer_group_id(0)
            pool = int(manager.get_mem_pool_base_address(LayerId(0), "key"))
            stride = manager.get_page_stride(LayerId(0), "key")
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                arena = int(batch.get_sparse_cache_base_address(LayerId(0), "key"))
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                # resize() bounds capacity - history to one token, so a unit is in flight
                # only as history reaches a unit boundary: two units of history behind the
                # watermark, and unit 2 in flight.
                self.assertTrue(cache.resize(3 * self.TPB, 3 * self.TPB - 1))
                batch.publish()

                out = torch.full((1, 1), 12345, dtype=torch.int32, device="cuda")
                batch.prepare_sparse_write_indices([BufferId(0, "key")], out, handle)
                torch.cuda.synchronize()
                index = int(out[0][0])
                self.assertNotEqual(index, BAD_PAGE_INDEX)

                written = self._pattern(self.BUF, 53)
                source = (ctypes.c_ubyte * self.BUF).from_buffer(bytearray(written))
                (err,) = cudart.cudaMemcpy(
                    arena + index * self.BUF,
                    ctypes.addressof(source),
                    self.BUF,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
                self.assertEqual(int(err), 0)

                # Paint the history pages differently, so a copy over the in-flight cell
                # would be visible rather than coincidentally matching.
                pages = [int(x) for x in cache.get_base_page_indices(lg)]
                for ordinal in range(2):
                    ctypes.memmove(
                        pool + pages[ordinal] * stride,
                        self._pattern(self.BUF, 60 + ordinal),
                        self.BUF,
                    )

                # A full-width selection of the two eligible units, so the fetch fills every
                # cell the reservation has beyond the one in flight.
                selected = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
                fetched = torch.full((1, 1, self.SELECTED), 12345, dtype=torch.int32, device="cuda")
                batch.fetch_sparse_pages([BufferId(0, "key")], selected, fetched, handle)
                torch.cuda.synchronize()

                after = torch.full((1, 1), 12345, dtype=torch.int32, device="cuda")
                batch.prepare_sparse_write_indices([BufferId(0, "key")], after, handle)
                torch.cuda.synchronize()
                self.assertEqual(int(after[0][0]), index, "the fetch moved the in-flight cell")

                got = bytearray(self.BUF)
                sink = (ctypes.c_ubyte * self.BUF).from_buffer(got)
                (err,) = cudart.cudaMemcpy(
                    ctypes.addressof(sink),
                    arena + index * self.BUF,
                    self.BUF,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
                self.assertEqual(int(err), 0)
                self.assertEqual(bytes(got), written, "the fetch overwrote the in-flight cell")
                # Removal returns the lease and discards the GPU cache, so the
                # write-back record has to match the history being given up.
                batch.flush_sparse_units([3 * self.TPB - 1])
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()


class TestSparseQuantisedDataPath(unittest.TestCase):
    """The quantised shape end to end: one layer, two buffers, two pools.

    Every other sparse data-path test runs a life cycle whose buffers are alike, so one pool
    serves them all and a single set of strides is right by accident. Here the data and its
    block scales differ in size, so they sit in pools -- and arena regions -- of their own,
    and the fetch is one launch per pool. A copy that used the wrong pool's cell size, slot
    stride or region base would read or write the wrong bytes, which is what this reads back.
    """

    TPB = 128
    DATA = 4096
    SCALE = 128
    SELECTED = 2
    CELLS = 4

    def setUp(self) -> None:
        init_cuda_once()

    @classmethod
    def _manager(cls) -> KVCacheManager:
        def buffer(role: str, size: int) -> BufferConfig:
            return BufferConfig(
                role,
                size,
                is_sparse=True,
            )

        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=cls.TPB,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 2)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            buffer("key", cls.DATA),
                            buffer("key_scale", cls.SCALE),
                            # A pool group sizes its slots from its device-backed pools.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=cls.SELECTED,
                            max_cached_units=cls.CELLS,
                            unit_tokens=cls.TPB,
                        ),
                    )
                ],
            )
        )

    def _deps(self):
        try:
            import ctypes

            import cuda.bindings.runtime as cudart
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch and the CUDA bindings are needed to address raw memory")
        return ctypes, cudart, torch

    @staticmethod
    def _pattern(size: int, seed: int) -> bytes:
        return bytes((i * seed + 13) % 251 for i in range(size))

    def test_each_buffer_is_fetched_into_its_own_pool_region(self):
        ctypes, cudart, torch = self._deps()
        manager = self._manager()
        cache = None
        roles = (("key", self.DATA, 7), ("key_scale", self.SCALE, 23))
        try:
            lg = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                self.assertTrue(cache.resize(4 * self.TPB, 4 * self.TPB))

                # Paint one block's hot page in each pool with a pattern of its own, so a
                # copy that crossed pools reads bytes that cannot pass for the right ones.
                ordinal = 1
                pages = [int(x) for x in cache.get_base_page_indices(lg)]
                patterns = {}
                for role, size, seed in roles:
                    patterns[role] = self._pattern(size, seed)
                    pool = int(manager.get_mem_pool_base_address(LayerId(0), role))
                    stride = manager.get_page_stride(LayerId(0), role)
                    ctypes.memmove(pool + pages[ordinal] * stride, patterns[role], size)

                batch.publish()
                selected = torch.tensor([[ordinal, -1]], dtype=torch.int32, device="cuda")
                out = torch.full((1, 2, self.SELECTED), 12345, dtype=torch.int32, device="cuda")
                batch.fetch_sparse_pages(
                    [BufferId(0, "key"), BufferId(0, "key_scale")],
                    selected,
                    out,
                    handle,
                )
                torch.cuda.synchronize()

                slots = batch.sparse_out_slots([BufferId(0, "key"), BufferId(0, "key_scale")])
                for position, (role, size, _) in enumerate(roles):
                    # At the buffer's slice, not its position: mirrored pools resolve to
                    # one index and share a slice, leaving the other holding the fill.
                    index = int(out[0][slots[position]][0])
                    self.assertNotEqual(index, 12345, f"{role}'s slice was never written")
                    self.assertNotEqual(index, BAD_PAGE_INDEX, role)
                    arena = int(batch.get_sparse_cache_base_address(LayerId(0), role))
                    got = bytearray(size)
                    sink = (ctypes.c_ubyte * size).from_buffer(got)
                    (err,) = cudart.cudaMemcpy(
                        ctypes.addressof(sink),
                        arena + index * size,
                        size,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )
                    self.assertEqual(int(err), 0, role)
                    self.assertEqual(
                        bytes(got), patterns[role], f"{role}'s cell does not hold its page"
                    )
                # Removal returns the lease and discards the GPU cache, so the
                # write-back record has to match the history being given up.
                batch.flush_sparse_units([4 * self.TPB])
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_each_buffer_is_written_back_to_its_own_hot_page(self):
        ctypes, cudart, torch = self._deps()
        manager = self._manager()
        cache = None
        roles = (("key", self.DATA, 31), ("key_scale", self.SCALE, 37))
        try:
            lg = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=1,
                    max_blocks_per_seq=8,
                    index_tables=manager.sparse_index_table_specs(),
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                # Capacity one token past a watermark that has not reached unit 0, so unit 0
                # is in flight and the resolve pins a cell for it in both pools.
                self.assertTrue(cache.resize(self.TPB, self.TPB - 1))
                batch.publish()
                selected = torch.tensor([[-1, -1]], dtype=torch.int32, device="cuda")
                out = torch.full((1, 2, self.SELECTED), 12345, dtype=torch.int32, device="cuda")
                batch.fetch_sparse_pages(
                    [BufferId(0, "key"), BufferId(0, "key_scale")],
                    selected,
                    out,
                    handle,
                )
                torch.cuda.synchronize()

                resident = _introspection.sparse_cache_resident_ordinals(cache, lg)
                self.assertIn(0, resident, "the in-flight unit must hold a cell")
                entry = resident.index(0)

                patterns = {}
                for pool_index, (role, size, seed) in enumerate(roles):
                    patterns[role] = self._pattern(size, seed)
                    arena = int(batch.get_sparse_cache_base_address(LayerId(0), role))
                    cell = _introspection.sparse_cache_base_cell(cache, lg, pool_index) + entry
                    source = (ctypes.c_ubyte * size).from_buffer(bytearray(patterns[role]))
                    (err,) = cudart.cudaMemcpy(
                        arena + cell * size,
                        ctypes.addressof(source),
                        size,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )
                    self.assertEqual(int(err), 0, role)

                batch.wait_published()
                # The watermark crosses unit 0, so both pools' cells are written back. The
                # flush runs before the move, since a block must reach its host page before
                # it becomes selectable.
                batch.flush_sparse_units([self.TPB])
                self.assertTrue(cache.resize(self.TPB + 1, self.TPB))
                torch.cuda.synchronize()

                pages = [int(x) for x in cache.get_base_page_indices(lg)]
                for role, size, _ in roles:
                    pool = int(manager.get_mem_pool_base_address(LayerId(0), role))
                    stride = manager.get_page_stride(LayerId(0), role)
                    landed = ctypes.string_at(pool + pages[0] * stride, size)
                    self.assertEqual(landed, patterns[role], f"{role}'s write-back went astray")
                batch.remove_kv_cache(cache)
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()


class TestSparseFlush(unittest.TestCase):
    """Writing a unit back to its host page once the watermark passes it."""

    def setUp(self) -> None:
        init_cuda_once()

    def test_only_units_the_watermark_crossed_are_listed(self):
        # Entries hold units 3, 7, 11 and one free entry.
        resident = [3, 7, 11, BAD_PAGE_INDEX]
        result = _introspection.sparse_flush_list(
            resident,
            num_layers=1,
            entries_per_layer=4,
            from_units=5,
            to_units=11,
            max_flush_units=4,
        )
        self.assertEqual(result.count, [1], "only unit 7 is in [5, 11)")
        self.assertEqual(
            result.total, [1], "the list is compacted, so its total says how far it is filled"
        )
        self.assertEqual(result.ordinal[0], 7)
        self.assertEqual(result.slot[0], 1)
        self.assertEqual(
            result.row[0], 0, "the entry carries its row, since position no longer does"
        )

    def test_a_watermark_that_moved_nothing_lists_nothing(self):
        resident = [3, 7, 11, BAD_PAGE_INDEX]
        result = _introspection.sparse_flush_list(
            resident, num_layers=1, entries_per_layer=4, from_units=8, to_units=8, max_flush_units=4
        )
        self.assertEqual(result.count, [0])

    def test_layers_are_scanned_independently(self):
        # Layer 0 holds unit 4, layer 1 holds unit 9; only layer 0's is in range.
        resident = [4, BAD_PAGE_INDEX, 9, BAD_PAGE_INDEX]
        result = _introspection.sparse_flush_list(
            resident, num_layers=2, entries_per_layer=2, from_units=0, to_units=5, max_flush_units=2
        )
        self.assertEqual(result.count, [1, 0])
        # One list per layer, each with its own counter, so layer 1's stays empty.
        self.assertEqual(result.total, [1, 0])
        self.assertEqual(result.ordinal[0], 4)

    def test_write_back_inverts_the_fetch(self):
        """Fetching a unit and writing it back must reproduce the cold page."""
        unit, units = 16, 2
        page_stride = unit * units
        cold = [(i * 5 + 3) % 251 for i in range(page_stride)]
        common = dict(
            host_page_stride=page_stride,
            page_table=[0],
            base_cell=0,
            num_blocks=1,
            cells_per_buffer=1,
            buffer_bytes=page_stride,
            num_chunks=1,
            chunk_bytes=unit,
            chunk_stride=0,
            unit_stride=unit,
            expansion=units,
            miss_ordinal=[1],
            miss_slot=[1],
            miss_count=[1],
            selection_width=1,
            buffer_layer=list(range(1)),
        )
        hot = _introspection.sparse_unit_copy(
            host_pool=cold, device_arena=[0] * page_stride, write_back=False, **common
        )
        self.assertEqual(hot[unit:], cold[unit:], "the fetch landed the unit")

        # Write that same entry back over a blanked cold page; the unit must return.
        restored = _introspection.sparse_unit_copy(
            host_pool=[0] * page_stride, device_arena=hot, write_back=True, **common
        )
        self.assertEqual(restored[unit:], cold[unit:], "the write-back restored the unit")
        self.assertEqual(restored[:unit], [0] * unit, "only the listed unit moved")

    def test_write_back_moves_into_the_cold_page(self):
        # The device pool is zeroed by the hook, so a write-back must blank the cold unit
        # it targets and leave the rest of the cold page alone.
        unit, units = 16, 2
        page_stride = unit * units
        cold = [(i * 5 + 3) % 251 for i in range(page_stride)]
        result = _introspection.sparse_unit_copy(
            host_pool=cold,
            host_page_stride=page_stride,
            page_table=[0],
            device_arena=[0] * (page_stride),
            base_cell=0,
            num_blocks=1,
            cells_per_buffer=1,
            buffer_bytes=page_stride,
            num_chunks=1,
            chunk_bytes=unit,
            chunk_stride=0,
            unit_stride=unit,
            expansion=units,
            miss_ordinal=[1],
            miss_slot=[1],
            miss_count=[1],
            selection_width=1,
            buffer_layer=list(range(1)),
            write_back=True,
        )
        self.assertEqual(result[unit:], [0] * unit, "unit 1 was overwritten from the entry")
        self.assertEqual(result[:unit], cold[:unit], "unit 0 was untouched")


class TestBatch(unittest.TestCase):
    """Row order and the device-side page table the sparse fetch reads."""

    BAD = BAD_PAGE_INDEX

    def setUp(self) -> None:
        init_cuda_once()

    @staticmethod
    def _manager() -> KVCacheManager:
        return KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=128,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                layers=[AttentionLayerConfig(0, [BufferConfig("key", 4096)])],
            )
        )

    def test_device_arrays_are_consumable_as_dlpack(self):
        """The page table and watermarks are what a caller actually receives.

        Imported lazily: the fast-mode path deliberately avoids loading the whole package.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to consume a DLPack array")

        manager = self._manager()
        try:
            batch = Batch(
                manager, NULL_STREAM, index_tables=[], max_batch_size=2, max_blocks_per_seq=4
            )
            lg_id = manager.get_layer_group_id(0)
            table = torch.from_dlpack(_introspection.batch_gpu_coalesced_page_table(batch, lg_id))
            history = torch.from_dlpack(batch.history_lengths())
            capacity = torch.from_dlpack(batch.capacities())

            self.assertEqual(tuple(table.shape), (2, 4))
            self.assertEqual(tuple(history.shape), (2,))
            self.assertEqual(tuple(capacity.shape), (2,))
            for arr in (table, history, capacity):
                self.assertEqual(arr.dtype, torch.int32)
                self.assertEqual(arr.device.type, "cuda")
            # Unoccupied rows: absent blocks, and nothing eligible.
            self.assertTrue(bool((table == BAD_PAGE_INDEX).all()))
            self.assertTrue(bool((history == 0).all()))
        finally:
            manager.shutdown()

    def test_sparse_leases_are_staged_per_row(self):
        """What is mirrored is the handle, not the memory: the reservation stays per request.

        A view appears for a member holding one and stays absent for a free row, so the
        fetch can tell the two apart.
        """
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=128,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                # The cache arena is reserved up front, so a sparse deployment must say
                # how many requests run together.
                constraints=[
                    BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 4)
                ],
                layers=[
                    AttentionLayerConfig(
                        0,
                        [
                            BufferConfig(
                                "key",
                                4096,
                                is_sparse=True,
                            ),
                            # A pool group is sized from its device-backed pools, so a sparse layer
                            # carries one. This is also the shape a model uses: an indexer and the
                            # data it selects over are one attention computation.
                            BufferConfig("index", 256),
                        ],
                        sparse_cache=SparseCacheConfig(
                            num_selected_blocks=2, max_cached_units=4, unit_tokens=128
                        ),
                    )
                ],
            )
        )
        cache = None
        batch = None
        try:
            lg_id = manager.get_layer_group_id(0)
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager,
                    handle,
                    max_batch_size=2,
                    max_blocks_per_seq=4,
                    index_tables=manager.sparse_index_table_specs(),
                )
                # Before anyone resumes, no reservation exists, so no view does either.
                batch.publish()
                self.assertEqual(_introspection.batch_sparse_entries_per_row(batch, lg_id), [0, 0])

                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                batch.publish()
                entries = _introspection.batch_sparse_entries_per_row(batch, lg_id)
                self.assertEqual(entries[0], manager.get_sparse_cache_max_units(0))
                self.assertEqual(entries[1], 0, "the free row stays absent")

                # Leaving releases the reservation, so the view must go with it rather than
                # leaving the fetch a handle into freed memory. A member is removed before
                # it suspends -- suspending in place would write its row behind the Batch's
                # back.
                batch.wait_published()
                batch.remove_kv_cache(cache)
                cache.suspend()
                batch.publish()
                self.assertEqual(_introspection.batch_sparse_entries_per_row(batch, lg_id), [0, 0])
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_membership_is_countable_and_a_hole_is_visible(self):
        """An engine mapping rows onto request slots asks the Batch, not a mirror of it."""
        manager = self._manager()
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=3, max_blocks_per_seq=4
                )
                self.assertEqual(batch.num_kv_caches, 0)
                self.assertTrue(batch.is_contiguous, "an empty batch has no hole")

                caches = [manager.create_kv_cache() for _ in range(3)]
                for cache in caches:
                    self.assertTrue(cache.resume(handle))
                    batch.add_kv_cache(cache)
                self.assertEqual(batch.num_kv_caches, 3)
                self.assertTrue(batch.is_contiguous)

                # A hole is legal, and is exactly what an engine wants to be told about.
                batch.remove_kv_cache(caches[1])
                self.assertEqual(batch.num_kv_caches, 2)
                self.assertFalse(batch.is_contiguous)

                # Refilling it closes the hole; dropping the tail never opens one.
                batch.add_kv_cache(caches[1])
                self.assertTrue(batch.is_contiguous)
                batch.remove_kv_cache(caches[2])
                self.assertEqual(batch.num_kv_caches, 2)
                self.assertTrue(batch.is_contiguous)

                for cache in caches:
                    if batch.row_of(cache) is not None:
                        batch.remove_kv_cache(cache)
                    cache.close()
        finally:
            manager.shutdown()

    def test_rows_are_slots_so_a_hole_is_reused(self):
        manager = self._manager()
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=3, max_blocks_per_seq=4
                )
                first = manager.create_kv_cache()
                second = manager.create_kv_cache()
                third = manager.create_kv_cache()
                for cache in (first, second, third):
                    self.assertTrue(cache.resume(handle))
                self.assertEqual(batch.add_kv_cache(first), 0)
                self.assertEqual(batch.add_kv_cache(second), 1)
                self.assertEqual(batch.add_kv_cache(third), 2)

                # Removing the middle member must not shift the ones after it: addresses and
                # row indices have to survive a graph capture.
                batch.remove_kv_cache(second)
                self.assertEqual(batch.row_of(third), 2)
                self.assertEqual(batch.row_of(second), None)

                fourth = manager.create_kv_cache()
                self.assertTrue(fourth.resume(handle))
                self.assertEqual(batch.add_kv_cache(fourth), 1, "the hole is reused")

                for cache in (first, third, fourth):
                    batch.remove_kv_cache(cache)
                for cache in (first, second, third, fourth):
                    cache.close()
        finally:
            manager.shutdown()

    def test_a_taken_row_is_refused_rather_than_overwritten(self):
        """Seating over a member would rebind its row, which nothing downstream would notice."""
        manager = self._manager()
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=2, max_blocks_per_seq=4
                )
                first = manager.create_kv_cache()
                second = manager.create_kv_cache()
                for cache in (first, second):
                    self.assertTrue(cache.resume(handle))
                self.assertEqual(batch.add_kv_cache(first, 0), 0)

                with self.assertRaisesRegex(Exception, "already occupied"):
                    batch.add_kv_cache(second, 0)
                with self.assertRaisesRegex(Exception, "beyond max_batch_size"):
                    batch.add_kv_cache(second, 2)
                # Neither attempt seated it, and neither disturbed the member already there.
                self.assertEqual(batch.row_of(second), None)
                self.assertEqual(batch.row_of(first), 0)
                self.assertEqual(batch.num_kv_caches, 1)

                # The row it was refused is still free to whoever asks for it properly.
                self.assertEqual(batch.add_kv_cache(second), 1)
                batch.remove_kv_cache(first)
                batch.remove_kv_cache(second)
                first.close()
                second.close()
        finally:
            manager.shutdown()

    def test_indexing_a_row_gives_its_member_or_none(self) -> None:
        """The inverse of row_of(). None is an empty row, never a bad index."""
        manager = self._manager()
        caches = []
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=3, max_blocks_per_seq=4
                )
                cache = manager.create_kv_cache()
                caches.append(cache)
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache, 1)

                seated = batch[1]
                self.assertIsNotNone(seated)
                # Compared through row_of rather than identity: whether nanobind hands back
                # the same Python object is its business, not this API's contract.
                self.assertEqual(batch.row_of(seated), 1)
                self.assertIsNone(batch[0], "a free row reads as empty")
                self.assertIsNone(batch[2])

                # Out of range is a different answer from empty, and must not be confusable.
                for bad in (3, -1, 99):
                    with self.assertRaises(IndexError):
                        batch[bad]
        finally:
            for cache in caches:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_move_carries_bindings_and_watermarks_to_the_empty_row(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - fast mode without torch
            raise unittest.SkipTest("torch is needed to pass device arrays")

        manager = self._manager()
        caches = []
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=3, max_blocks_per_seq=4
                )
                lg = manager.get_layer_group_id(0)
                table = torch.from_dlpack(_introspection.batch_gpu_coalesced_page_table(batch, lg))
                address = table.data_ptr()
                for history in (17, 31):
                    cache = manager.create_kv_cache()
                    caches.append(cache)
                    self.assertTrue(cache.resume(handle))
                    self.assertTrue(cache.resize(128, history))
                    batch.add_kv_cache(cache)
                batch.flush_sparse_units([17, 31, 0])
                batch.publish()

                # Row 1 to the free row 2. Row 0 is a bystander and must not move.
                batch.move(2, 1)
                self.assertEqual([batch.row_of(cache) for cache in caches], [0, 2])
                self.assertEqual(batch.flushed_history(2), 31, "the watermark travels")
                self.assertEqual(batch.flushed_history(0), 17, "the bystander is untouched")

                batch.publish()
                torch.cuda.synchronize()
                self.assertEqual(table.data_ptr(), address, "device tables keep their addresses")
                self.assertEqual(table[0].tolist(), list(caches[0].get_base_page_indices(lg)))
                self.assertEqual(table[2].tolist(), list(caches[1].get_base_page_indices(lg)))
                self.assertTrue(
                    bool((table[1] == BAD_PAGE_INDEX).all()),
                    "the vacated row publishes as free, not as a stale copy of the member",
                )
                self.assertEqual(torch.from_dlpack(batch.history_lengths()).tolist(), [17, 0, 31])
                self.assertEqual(torch.from_dlpack(batch.capacities()).tolist(), [128, 0, 128])
        finally:
            for cache in caches:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_move_rejects_a_bad_pair_without_changing_rows(self) -> None:
        manager = self._manager()
        caches = []
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=3, max_blocks_per_seq=4
                )
                for _ in range(2):
                    cache = manager.create_kv_cache()
                    caches.append(cache)
                    self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(caches[0], 0)
                batch.add_kv_cache(caches[1], 1)

                # Matched on the message rather than the type, so each case is pinned to the
                # reason it is refused: a type alone would not distinguish them.
                #
                # (dst, src). dst == src is both faults at once, and is reported as the
                # occupied destination -- a row cannot be occupied and empty together.
                rejected = {
                    (0, 1): "destination row is already occupied",
                    (1, 1): "destination row is already occupied",
                    (0, 0): "destination row is already occupied",
                    (1, 2): "source row holds no member",
                    (3, 0): "beyond max_batch_size",
                    (2, 3): "beyond max_batch_size",
                    (-1, 0): "beyond max_batch_size",
                    (2, -1): "beyond max_batch_size",
                }
                for (dst, src), message in rejected.items():
                    with self.assertRaisesRegex(Exception, message):
                        batch.move(dst, src)
                    self.assertEqual([batch.row_of(cache) for cache in caches], [0, 1])

                # Still usable afterwards: a refusal must not have half-moved anything.
                batch.move(2, 1)
                self.assertEqual([batch.row_of(cache) for cache in caches], [0, 2])
        finally:
            for cache in caches:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_the_batch_keeps_a_seated_member_alive(self):
        """A caller may keep only the Batch: seating is what holds the cache up.

        Dropping the caller's reference and then getting the cache back from the Batch is
        the whole contract -- if the row held a raw pointer the object would already be
        gone, and what came back would name freed memory.
        """
        manager = self._manager()
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=2, max_blocks_per_seq=4
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                row = batch.add_kv_cache(cache)

                # Drop the caller's only reference. Were the row raw, this would destroy the
                # cache and poison on the way out, since a member must not outlive its row.
                del cache
                self.assertEqual(batch.num_kv_caches, 1)

                # Leaving hands it back, still usable -- which a dangling row could not do.
                returned = batch.remove_row(row)
                self.assertIsNotNone(returned)
                self.assertEqual(batch.num_kv_caches, 0)
                self.assertEqual(batch.row_of(returned), None)
                returned.close()

                # A row nobody occupies has nothing to hand back.
                self.assertIsNone(batch.remove_row(0))
        finally:
            manager.shutdown()

    def test_membership_is_exclusive(self):
        manager = self._manager()
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                one = Batch(
                    manager, handle, index_tables=[], max_batch_size=2, max_blocks_per_seq=4
                )
                two = Batch(
                    manager, handle, index_tables=[], max_batch_size=2, max_blocks_per_seq=4
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                one.add_kv_cache(cache)
                with self.assertRaisesRegex(Exception, "at most one Batch"):
                    two.add_kv_cache(cache)
                one.remove_kv_cache(cache)
                # Free to join another batch once released.
                two.add_kv_cache(cache)
                two.remove_kv_cache(cache)
                cache.close()
        finally:
            manager.shutdown()

    def test_beam_search_is_refused_rather_than_half_mirrored(self):
        """Only beam 0 is staged, so a wider request would have every beam read beam 0."""
        # Partial commit forbids beam_width changes, so this manager disables it.
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=128,
                cache_tiers=[GpuCacheTierConfig(64 << 20), HostCacheTierConfig(64 << 20)],
                layers=[AttentionLayerConfig(0, [BufferConfig("key", 4096)])],
                enable_partial_commit=False,
            )
        )
        cache = None
        try:
            batch = Batch(
                manager, NULL_STREAM, index_tables=[], max_batch_size=1, max_blocks_per_seq=4
            )
            cache = manager.create_kv_cache()
            cache.beam_width = 2
            with self.assertRaisesRegex(Exception, "beam"):
                batch.add_kv_cache(cache)
            self.assertEqual(batch.row_of(cache), None, "a refused member takes no row")
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_a_full_batch_refuses_rather_than_growing(self):
        manager = self._manager()
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=1, max_blocks_per_seq=4
                )
                first = manager.create_kv_cache()
                second = manager.create_kv_cache()
                self.assertTrue(first.resume(handle))
                self.assertTrue(second.resume(handle))
                batch.add_kv_cache(first)
                # Growing would move the device arrays, which a captured graph cannot tolerate.
                with self.assertRaisesRegex(Exception, "full"):
                    batch.add_kv_cache(second)
                batch.remove_kv_cache(first)
                first.close()
                second.close()
        finally:
            manager.shutdown()

    def test_an_unoccupied_row_reads_as_absent(self):
        manager = self._manager()
        try:
            lg_id = manager.get_layer_group_id(0)
            with TemporaryCudaStream([]) as stream:
                batch = Batch(
                    manager,
                    cast(CudaStream, stream.handle),
                    index_tables=[],
                    max_batch_size=2,
                    max_blocks_per_seq=3,
                )
                batch.publish()
                table = _introspection.batch_page_table(batch, lg_id)
            # Every cell is the sentinel, not slot zero: only the sentinel distinguishes
            # an absent block from a real page.
            self.assertEqual(table, [self.BAD] * 6)
        finally:
            manager.shutdown()

    def test_publish_mirrors_a_members_page_indices(self):
        manager = self._manager()
        cache = None
        batch = None
        try:
            lg_id = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=2, max_blocks_per_seq=4
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                cache.resize(256, 0)  # 128-token blocks, so two of them
                batch.publish()
                table = _introspection.batch_page_table(batch, lg_id)

                # The installed buffer is max_blocks_per_seq long, so the member reports
                # that whole span: its blocks first, then the sentinel tail.
                row = list(cache.get_base_page_indices(lg_id))
                self.assertEqual(len(row), 4)
                self.assertEqual(table[:4], row, "row 0 mirrors the member")
                self.assertTrue(all(i != self.BAD for i in row[:2]), row)
                self.assertEqual(row[2:], [self.BAD] * 2, "past the member's blocks")
                # Row 1 is unoccupied and stays absent.
                self.assertEqual(table[4:], [self.BAD] * 4)
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_batch_resize_advances_every_member_and_publishes(self):
        manager = self._manager()
        caches = []
        batch = None
        try:
            lg_id = manager.get_layer_group_id(0)
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=3, max_blocks_per_seq=8
                )
                for _ in range(2):
                    cache = manager.create_kv_cache()
                    caches.append(cache)
                    self.assertTrue(cache.resume(handle))
                    batch.add_kv_cache(cache)

                results = batch.resize([256, 384, None], [0, 0, None])
                # Row 2 is free, so it reports False rather than raising.
                self.assertEqual(results, [True, True, False])
                self.assertEqual(caches[0].capacity, 256)
                self.assertEqual(caches[1].capacity, 384)

                # Published as part of the resize, so no separate publish is needed.
                table = _introspection.batch_page_table(batch, lg_id)
                self.assertEqual(sum(1 for i in table[0:8] if i != self.BAD), 2)
                self.assertEqual(sum(1 for i in table[8:16] if i != self.BAD), 3)
                self.assertEqual(table[16:24], [self.BAD] * 8, "the free row stays absent")
        finally:
            for cache in caches:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_watermarks_publish_with_the_page_table(self):
        """One transfer carries both, so a kernel can never pair mismatched steps."""
        manager = self._manager()
        caches = []
        batch = None
        try:
            lg_id = manager.get_layer_group_id(0)
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=3, max_blocks_per_seq=8
                )
                # Seeded before joining: the Batch only accepts a history its row already
                # had the capacity for, and these start at nothing.
                for capacity, history in ((256, 128), (384, 256)):
                    cache = manager.create_kv_cache()
                    caches.append(cache)
                    self.assertTrue(cache.resume(handle))
                    self.assertTrue(cache.resize(capacity, history))
                    batch.add_kv_cache(cache)

                batch.resize([256, 384, None], [128, 256, None])
                history, capacity = _introspection.batch_watermarks(batch)
                self.assertEqual(history, [128, 256, 0])
                self.assertEqual(capacity, [256, 384, 0], "a free row reads as nothing eligible")

                # num_blocks is derived, not published: history_length // tokens_per_block.
                table = _introspection.batch_page_table(batch, lg_id)
                for row, hist in enumerate(history):
                    num_blocks = hist // 128
                    valid = table[row * 8 : row * 8 + num_blocks]
                    self.assertTrue(
                        all(i != self.BAD for i in valid),
                        f"row {row} must have {num_blocks} valid entries, got {valid}",
                    )
        finally:
            for cache in caches:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_a_removed_member_leaves_no_watermark_behind(self):
        manager = self._manager()
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=1, max_blocks_per_seq=4
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                # Seeded before joining: the Batch only accepts a history its row already
                # had the capacity for.
                self.assertTrue(cache.resize(256, 128))
                batch.add_kv_cache(cache)
                batch.resize([256], [128])
                self.assertEqual(_introspection.batch_watermarks(batch)[0], [128])

                batch.remove_kv_cache(cache)
                batch.publish()
                # Otherwise the row would still claim blocks the departed request owned.
                history, capacity = _introspection.batch_watermarks(batch)
                self.assertEqual((history, capacity), ([0], [0]))
        finally:
            if cache is not None:
                cache.close()
            manager.shutdown()

    def test_a_joining_member_publishes_its_own_watermarks(self):
        """A row holds whatever its last occupant left until the new one seeds it."""
        manager = self._manager()
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=1, max_blocks_per_seq=8
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                # Advanced while unbatched, which is what makes the row's own state stale.
                self.assertTrue(cache.resize(256, 128))

                batch.add_kv_cache(cache)
                batch.publish()
                self.assertEqual(
                    _introspection.batch_watermarks(batch),
                    ([128], [256]),
                    "the row must describe the member that now occupies it",
                )
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_batch_resize_needs_one_entry_per_row(self):
        manager = self._manager()
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=2, max_blocks_per_seq=4
                )
                # Indexed by row, so a short list would silently misalign members.
                with self.assertRaisesRegex(Exception, "per row"):
                    batch.resize([128], [0])
        finally:
            manager.shutdown()

    def test_history_may_not_outrun_the_capacity_the_row_already_had(self):
        """Capacity granted now makes room for tokens written later, not already written."""
        manager = self._manager()
        cache = None
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=1, max_blocks_per_seq=8
                )
                cache = manager.create_kv_cache()
                self.assertTrue(cache.resume(handle))
                # Seeded before joining: the Batch only accepts a history its row already
                # had the capacity for.
                self.assertTrue(cache.resize(256, 128))
                batch.add_kv_cache(cache)
                batch.resize([256], [128])

                # Capacity 256 now, so token 256 has no space yet and cannot be history,
                # however much capacity this same call would add.
                with self.assertRaisesRegex(Exception, "capacity the row already had"):
                    batch.resize([512], [384])
                # Refused before anything moved.
                self.assertEqual(_introspection.batch_watermarks(batch), ([128], [256]))

                with self.assertRaisesRegex(Exception, "cannot be decreased"):
                    batch.resize([256], [64])
                self.assertEqual(_introspection.batch_watermarks(batch), ([128], [256]))
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_a_row_whose_growth_fails_still_advances_its_history(self):
        """The advance needs no slots, and is what frees the cells the growth wanted."""
        tpb, page = 128, 4096
        # The pool bottoms out at 512 blocks whatever the quota, so two rows of 256 fill
        # it and a third-party grow to 512 has nowhere to come from. Measured, not derived
        # from the quota: a small quota does not shrink the pool below that floor.
        blocks, pool = 256, 512
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=tpb,
                cache_tiers=[GpuCacheTierConfig(pool * page)],
                layers=[AttentionLayerConfig(0, [BufferConfig("key", page)])],
            )
        )
        caches = []
        batch = None
        try:
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=2, max_blocks_per_seq=pool
                )
                for _ in range(2):
                    cache = manager.create_kv_cache()
                    caches.append(cache)
                    self.assertTrue(cache.resume(handle))
                    batch.add_kv_cache(cache)

                each = blocks * tpb
                self.assertEqual(list(batch.resize([each, each], [0, 0])), [1, 1])

                # The two rows now hold the whole pool, so row 1's grow has nowhere to come
                # from and rolls back. Row 0 only advances history, which never allocates.
                results = list(batch.resize([each, pool * tpb], [each, each]))
                self.assertEqual(results[0], 1)
                self.assertEqual(
                    results[1], 0, "the grow must actually fail or this proves nothing"
                )

                history, capacity = _introspection.batch_watermarks(batch)
                self.assertEqual(history, [each, each], "both advanced; neither needed a slot")
                self.assertEqual(capacity, [each, each], "row 1's growth rolled itself back")
        finally:
            for cache in caches:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_wait_published_is_a_noop_when_nothing_is_in_flight(self):
        manager = self._manager()
        try:
            batch = Batch(
                manager, NULL_STREAM, index_tables=[], max_batch_size=1, max_blocks_per_seq=4
            )
            # Safe to call at the top of a step whether or not a publish happened.
            batch.wait_published()
            batch.wait_published()
        finally:
            manager.shutdown()

    def test_a_republish_after_waiting_shows_the_new_indices(self):
        """A republish after waiting must carry the change.

        publish() copies asynchronously from pinned staging, so the host has to wait
        before touching a member again.
        """
        manager = self._manager()
        cache = None
        batch = None
        try:
            lg_id = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=1, max_blocks_per_seq=8
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                cache.resize(256, 0)
                batch.publish()
                first = _introspection.batch_page_table(batch, lg_id)
                self.assertEqual(sum(1 for i in first if i != self.BAD), 2)

                # The contract: wait before the resize, which writes the staging row the
                # copy above is still reading.
                batch.wait_published()
                cache.resize(512, 0)
                batch.publish()
                second = _introspection.batch_page_table(batch, lg_id)
                self.assertEqual(sum(1 for i in second if i != self.BAD), 4, second)
        finally:
            if cache is not None:
                if batch is not None and batch.row_of(cache) is not None:
                    batch.remove_kv_cache(cache)
                cache.close()
            manager.shutdown()

    def test_a_removed_members_row_is_cleared(self):
        manager = self._manager()
        try:
            lg_id = manager.get_layer_group_id(0)
            cache = manager.create_kv_cache()
            with TemporaryCudaStream([]) as stream:
                handle = cast(CudaStream, stream.handle)
                batch = Batch(
                    manager, handle, index_tables=[], max_batch_size=1, max_blocks_per_seq=4
                )
                self.assertTrue(cache.resume(handle))
                batch.add_kv_cache(cache)
                cache.resize(256, 0)
                batch.remove_kv_cache(cache)
                batch.publish()
                # Otherwise the row would still name pages the departed request owns.
                self.assertEqual(_introspection.batch_page_table(batch, lg_id), [self.BAD] * 4)
        finally:
            cache.close()
            manager.shutdown()


class TestKVCacheManagerV2(unittest.TestCase):
    engine: FakeEngine
    cfg: KVCacheManagerConfig
    manager: KVCacheManager
    _token_id_gen: Iterator[int]

    def setUp(self) -> None:
        init_cuda_once()
        self._token_id_gen = itertools.count()
        gc.collect()
        gc.disable()

    def tearDown(self) -> None:
        gc.enable()
        if hasattr(self, "manager"):
            # Drop the manager even if shutdown() raises, which it does when a test leaves a KV
            # cache open. Holding on to it would keep the poison latch un-clearable and fail
            # every later test instead of just this one.
            try:
                self.manager.shutdown()
            finally:
                del self.manager

    def next_token(self) -> TokenIdExt:
        token_id = next(self._token_id_gen)
        if token_id % 100 == 99:
            return randbytes(32)
        else:
            return TokenId(token_id)

    def prepare(
        self,
        gpu_quota: int,
        host_quota: int,
        disk_quota: int,
        num_layers: int,
        window_size: SlidingWindowSize,
        sink_tokens: int,
        tokens_per_block: int = 32,
        kv_buf_size: int = 8192,
        block_quant_buf_size: int | None = None,
    ) -> None:
        self.cfg = create_config(
            tokens_per_block,
            gpu_quota,
            host_quota,
            disk_quota,
            num_layers,
            window_size,
            sink_tokens,
            kv_buf_size,
            block_quant_buf_size,
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)


class TestRetention(TestKVCacheManagerV2):
    def setUp(self) -> None:
        super().setUp()
        self.caches: list[_KVCache] = []
        self.stream = CachedCudaStream()

    def tearDown(self) -> None:
        self._close_caches()
        super().tearDown()

    def _close_caches(self) -> None:
        for cache in reversed(self.caches):
            cache.close()
        self.caches.clear()

    def _prepare_retention(
        self,
        *,
        host_quota: int = 0,
        window_size: SlidingWindowSize = None,
        enable_partial_commit: bool = True,
        offload_threshold: int = DEFAULT_COLD_TIER_OFFLOAD_MIN_PRIORITY,
    ) -> None:
        self.cfg = create_config(4, 2 << 20, host_quota, 0, 1, window_size, 0, 65536)
        self.cfg.enable_partial_commit = enable_partial_commit
        self.cfg.cold_tier_offload_min_priority = offload_threshold
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

    def _new_cache(
        self,
        tokens: list[TokenIdExt],
        config: KvCacheRetentionConfig | None = None,
        prompt_length: int | None = None,
    ) -> _KVCache:
        cache = self.manager.create_kv_cache(
            input_tokens=tokens,
            expected_prompt_length=len(tokens) if prompt_length is None else prompt_length,
            kv_cache_retention_config=config,
        )
        self.caches.append(cache)
        return cache

    def _cache(
        self,
        tokens: list[TokenIdExt],
        config: KvCacheRetentionConfig | None = None,
        prompt_length: int | None = None,
    ) -> _KVCache:
        cache = self._new_cache(tokens, config, prompt_length)
        self.assertTrue(cache.resume(self.stream.handle))
        self.assertTrue(cache.resize(len(tokens)))
        remaining = tokens[cache.num_committed_tokens :]
        if tokens:
            self.engine.execute(
                [Step(cache, remaining, tokens[: cache.num_committed_tokens])], self.stream.handle
            )
        if remaining:
            cache.commit(remaining)
        return cache

    def _apply_memory_pressure(self) -> _KVCache:
        free = self.manager.get_storage_statistics(GPU_LEVEL)[0].free
        pressure = self._cache([])
        self.assertTrue(pressure.resize((free + 1) * self.cfg.tokens_per_block))
        return pressure

    @staticmethod
    def _config(
        priority: int | None, duration: timedelta | None = None
    ) -> KvCacheRetentionConfig | None:
        if priority is None:
            return None
        return KvCacheRetentionConfig(
            [KvCacheRetentionConfig.TokenRangeRetentionConfig(0, None, priority, duration)]
        )

    def test_page_priorities_reject_invalid_layer_groups(self) -> None:
        self._prepare_retention()
        for num_tokens in (0, 4):
            cache = self._cache(list(range(num_tokens)))
            for layer_group in (-1, 1):
                with self.subTest(num_tokens=num_tokens, layer_group=layer_group):
                    with self.assertRaises(IndexError):
                        cache.get_page_priorities(layer_group)
            self.assertEqual(cache.get_page_priorities(0), [35] * (num_tokens // 4))

    @parameterized.expand([(4, False), (4, True), (5, False), (5, True)])
    def test_beam_priorities_share_prompt_and_keep_private_tail(
        self, prompt_length: int, commit: bool
    ) -> None:
        self._prepare_retention(enable_partial_commit=False)
        config = KvCacheRetentionConfig(
            [KvCacheRetentionConfig.TokenRangeRetentionConfig(0, None, 80)],
            decode_retention_priority=10,
        )
        tokens = list(range(prompt_length))
        cache = self._new_cache(tokens, config)
        self.assertTrue(cache.resume(self.stream.handle))
        self.assertTrue(cache.resize(12))
        self.engine.execute([Step(cache, tokens, [])], self.stream.handle)
        if commit:
            cache.commit(tokens)
        cache.stop_committing()
        cache.beam_width = 3

        prompt_blocks = div_up(prompt_length, 4)
        shared_blocks = prompt_length // 4
        for capacity in (12, 16):
            self.assertTrue(cache.resize(capacity))
            indices = [cache.get_aggregated_page_indices(0, beam) for beam in range(3)]
            for beam in range(3):
                self.assertEqual(
                    cache.get_page_priorities(0, beam),
                    [80] * prompt_blocks + [10] * (capacity // 4 - prompt_blocks),
                )
                if beam:
                    self.assertEqual(
                        indices[beam][:shared_blocks], [BAD_PAGE_INDEX] * shared_blocks
                    )
            # The partial prompt tail and both preallocated and newly allocated decode pages
            # are writable independently by every beam.
            for ordinal in range(shared_blocks, capacity // 4):
                pages = {row[ordinal] for row in indices}
                self.assertNotIn(BAD_PAGE_INDEX, pages)
                self.assertEqual(len(pages), 3)
        for invalid_beam in (-1, 3):
            with self.assertRaises(IndexError):
                cache.get_page_priorities(0, invalid_beam)
        self.stream.synchronize()

    def test_beam_priorities_read_each_private_page(self) -> None:
        self._prepare_retention(enable_partial_commit=False)
        priority = 60
        tokens = list(range(5))
        cache = self.manager.create_kv_cache(
            input_tokens=tokens,
            expected_prompt_length=len(tokens),
            custom_priority_callback=lambda ordinal, lifecycle: priority,
        )
        self.caches.append(cache)
        self.assertTrue(cache.resume(self.stream.handle))
        self.assertTrue(cache.resize(12))
        self.engine.execute([Step(cache, tokens, [])], self.stream.handle)
        cache.stop_committing()
        # Give each new beam's private pages a distinct allocation-time priority.
        for beam_width, priority in ((2, 80), (3, 90)):
            cache.beam_width = beam_width
        self.assertEqual(cache.get_page_priorities(0, 0), [60, 60, 60])
        self.assertEqual(cache.get_page_priorities(0, 1), [60, 80, 80])
        self.assertEqual(cache.get_page_priorities(0, 2), [60, 90, 90])
        self.stream.synchronize()

    def test_offload_default_and_dataclass_replace(self) -> None:
        config = create_config(4, 2 << 20, 0, 0, 1, None, 0, 65536)
        self.assertEqual(
            config.cold_tier_offload_min_priority, DEFAULT_COLD_TIER_OFFLOAD_MIN_PRIORITY
        )
        self.assertEqual(
            replace(config).cold_tier_offload_min_priority, DEFAULT_COLD_TIER_OFFLOAD_MIN_PRIORITY
        )
        for threshold in (0, 80):
            self.assertEqual(
                replace(
                    config, cold_tier_offload_min_priority=threshold
                ).cold_tier_offload_min_priority,
                threshold,
            )

    @parameterized.expand(
        [(80, 10, False), (80, 10, True), (10, 80, False), (10, 80, True), (80, None, False)]
    )
    def test_shared_claim_order_controls_eviction(
        self, first: int, second: int | None, close_newer_first: bool
    ) -> None:
        self._prepare_retention()
        prompt = [1, 2, 3, 4]
        late = self._new_cache(prompt, self._config(100))
        self.assertTrue(late.resume(self.stream.handle))
        self.assertTrue(late.resize(len(prompt)))
        self.engine.execute([Step(late, prompt, [])], self.stream.handle)
        older = self._cache(prompt, self._config(first))
        newer = self._cache(prompt, self._config(second))
        priority = first if second is None else second
        # Deduplicating an already computed page is not a new request claim.
        late.commit(prompt)
        self.assertEqual(late.get_page_priorities(0), [priority])
        late.close()
        self.assertEqual(older.get_page_priorities(0), [priority])
        self.assertEqual(newer.get_aggregated_page_indices(0), older.get_aggregated_page_indices(0))
        requests = [newer, older] if close_newer_first else [older, newer]
        requests[0].close()
        self.assertEqual(requests[1].get_page_priorities(0), [priority])
        competitor = self._cache([5, 6, 7, 8], self._config(50))
        competitor.close()
        # Referenced pages cannot be permanently evicted, irrespective of priority.
        pin = self._apply_memory_pressure()
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 4)
        self.assertEqual(self.manager.probe_reuse(input_tokens=[5, 6, 7, 8]), 0)
        pin.close()
        requests[1].close()
        competitor = self._cache([5, 6, 7, 8], self._config(50))
        competitor.close()
        self._apply_memory_pressure()
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 4 if priority > 50 else 0)
        self.assertEqual(
            self.manager.probe_reuse(input_tokens=[5, 6, 7, 8]), 0 if priority > 50 else 4
        )

    def test_prompt_and_decode_priorities_expire_after_last_release(self) -> None:
        self._prepare_retention()
        duration = timedelta(milliseconds=20)
        config = KvCacheRetentionConfig(
            [KvCacheRetentionConfig.TokenRangeRetentionConfig(1, None, 80, duration)],
            decode_retention_priority=10,
            decode_duration_ms=duration,
        )
        tokens = list(range(12))
        # A five-token prompt occupies two blocks; its partial second block stays context.
        first = self._cache(tokens[:5], config)
        self.assertTrue(first.resize(len(tokens)))
        self.engine.execute([Step(first, tokens[5:], tokens[:5])], self.stream.handle)
        first.commit(tokens[5:])
        second = self._cache(tokens, config, prompt_length=5)
        self.assertEqual(first.get_page_priorities(0), [35, 80, 10])
        first.close()
        time.sleep(0.04)
        self.manager.refresh_retention()
        self.assertEqual(second.get_page_priorities(0), [35, 80, 10])
        second.close()
        time.sleep(0.04)
        self.assertTrue(self.manager.prefetch_reuse(ReuseScope(), tokens, GPU_LEVEL))
        self.manager.refresh_retention()
        reclaimed = self._cache(tokens)
        self.assertEqual(reclaimed.get_page_priorities(0), [35, 35, 35])

    def test_unconfigured_claim_preserves_priority_and_clears_expiry(self) -> None:
        self._prepare_retention()
        tokens = list(range(12))
        long_duration = timedelta(seconds=1)
        config = KvCacheRetentionConfig(
            [KvCacheRetentionConfig.TokenRangeRetentionConfig(1, None, 80, long_duration)],
            decode_retention_priority=10,
            decode_duration_ms=long_duration,
        )
        timed = self._cache(tokens, config, prompt_length=5)
        timed.close()
        untimed = self._cache(tokens)
        untimed.close()
        time.sleep(1.05)
        self.manager.refresh_retention()
        self.assertEqual(self._cache(tokens).get_page_priorities(0), [35, 80, 10])

    @parameterized.expand([(False,), (True,)])
    def test_sliding_window_release_preserves_priority_without_expiry(self, shared: bool) -> None:
        self._prepare_retention(window_size=4)
        tokens = list(range(8))
        config = self._config(80, timedelta(milliseconds=20))
        source = self._cache(tokens[:4], config)
        sharer = self._cache(tokens[:4], config) if shared else None
        self.assertTrue(source.resize(8))
        self.engine.execute([Step(source, tokens[4:], tokens[:4])], self.stream.handle)
        source.commit(tokens[4:])
        source.close()
        if sharer is not None:
            sharer.close()
        time.sleep(0.04)
        self.manager.refresh_retention()
        self.assertEqual(self._cache(tokens[:4]).get_page_priorities(0), [80])

    def test_prefetch_rejects_invalid_cache_levels(self) -> None:
        self._prepare_retention()
        for level in (-1, 1):
            with self.subTest(level=level), self.assertRaises(IndexError):
                self.manager.prefetch_reuse(ReuseScope(), [1, 2, 3, 4], CacheLevel(level))

    def test_offload_reload_preserves_priority_and_partial_reuse(self) -> None:
        self._prepare_retention(host_quota=2 << 20)
        tokens = [1, 2, 3, 4]
        cached = self._cache(tokens, self._config(80))
        cached.suspend()
        pressure = self._apply_memory_pressure()
        self.assertEqual(_introspection.active_page_stats(cached)[0], [0, 1])
        pressure.close()
        self.assertTrue(cached.resume(self.stream.handle))
        self.assertEqual(cached.get_page_priorities(0), [80])
        # The resume brought the page back: active on the hot level, gone from host.
        self.assertEqual(_introspection.active_page_stats(cached)[0], [1, 0])
        self.engine.execute([Step(cached, [], tokens)], self.stream.handle)
        self.stream.synchronize()
        for config in (None, self._config(10)):
            partial = self._cache(tokens[:3], config)
            self.assertEqual(partial.num_committed_tokens, 3)
            self.assertEqual(partial.get_page_priorities(0), [80])
            self.assertEqual(cached.get_page_priorities(0), [80])
            self.assertNotEqual(
                partial.get_aggregated_page_indices(0), cached.get_aggregated_page_indices(0)
            )
        self._close_caches()
        self.manager.clear_reusable_blocks()
        for level in (GPU_LEVEL, CacheLevel(1)):
            for stats in self.manager.get_storage_statistics(level):
                self.assertEqual(stats.free, stats.total)
                self.assertEqual(stats.evictable, 0)

    def test_failed_resize_preserves_capacity_priority_and_data(self) -> None:
        self._prepare_retention(host_quota=2 << 20)
        tokens = [1, 2, 3, 4]
        cached = self._cache(tokens, self._config(80))
        capacity = cached.capacity
        total = self.manager.get_storage_statistics()[0].total
        self.assertFalse(cached.resize((total + 1) * 4))
        self.assertEqual(cached.capacity, capacity)
        self.assertEqual(cached.get_page_priorities(0), [80])
        self.engine.execute([Step(cached, [], tokens)], self.stream.handle)
        self.stream.synchronize()

    def test_unsupported_transfer_modes_do_not_claim_reusable_pages(self) -> None:
        self._prepare_retention(host_quota=2 << 20)
        tokens = [1, 2, 3, 4]
        self._cache(tokens, self._config(80)).close()
        # Unsupported file-transfer semantics fail before claiming any reusable pages.
        for mode in (KvCacheTransferMode.GDS, KvCacheTransferMode.POSIX_DEBUG_FALLBACK):
            config = KvCacheRetentionConfig([], transfer_mode=mode, directory="/tmp")
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "per-request GDS/POSIX"),
            ):
                self.manager.create_kv_cache(input_tokens=tokens, kv_cache_retention_config=config)
        self.assertEqual(self._cache(tokens).get_page_priorities(0), [80])

    def test_deferred_partial_copies_use_latest_claim_priority(self) -> None:
        self._prepare_retention(host_quota=2 << 20)
        donor = self._cache(list(range(8)), self._config(80))
        donor.close()
        pending = [self._new_cache(list(range(3)), self._config(priority)) for priority in (10, 90)]
        for copy in pending:
            self.assertTrue(copy.resume(self.stream.handle))
            self.assertEqual(copy.get_page_priorities(0), [90])
            self.engine.execute([Step(copy, [], list(range(3)))], self.stream.handle)
        self.stream.synchronize()

    @parameterized.expand([(29, 30), (30, 30), (49, 50), (50, 50), (0, -1), (100, 101)])
    def test_released_offload_threshold(self, priority: int, threshold: int) -> None:
        self._prepare_retention(host_quota=2 << 20, offload_threshold=threshold)
        tokens = [1, 2, 3, 4]
        cached = self._cache(tokens, self._config(priority, timedelta(seconds=2)))
        cached.close()
        pressure = self._apply_memory_pressure()
        retained = priority >= threshold
        self.assertEqual(self.manager.probe_reuse(input_tokens=tokens), 4 if retained else 0)
        self.assertEqual(
            self.manager.get_storage_statistics(CacheLevel(1))[0].evictable, 1 if retained else 0
        )
        pressure.close()
        self.assertEqual(self._cache(tokens).get_page_priorities(0), [priority if retained else 35])

    def test_eviction_offload_preserves_priority_and_clears_expiry(self) -> None:
        self._prepare_retention(host_quota=2 << 20, offload_threshold=50)
        tokens = [1, 2, 3, 4]
        self._cache(tokens, self._config(80, timedelta(seconds=2))).close()
        pressure = self._apply_memory_pressure()
        self.assertEqual(self.manager.probe_reuse(input_tokens=tokens), 4)
        self.assertEqual(self.manager.get_storage_statistics(CacheLevel(1))[0].evictable, 1)
        pressure.close()
        time.sleep(2.05)
        self.manager.refresh_retention()
        # GPU eviction-driven offload clears the old duration, retaining the priority.
        self.assertEqual(self._cache(tokens).get_page_priorities(0), [80])


class TestStorageStatistics(TestKVCacheManagerV2):
    def test_statistics_are_detached_snapshots(self) -> None:
        self.prepare(8 << 20, 8 << 20, 0, 2, 16, 0, tokens_per_block=4, kv_buf_size=4096)

        def values(stats):
            return [
                (
                    item.slot_sizes,
                    item.total,
                    item.free,
                    item.evictable,
                    item.available,
                    item.unavailable,
                )
                for item in stats
            ]

        snapshots = [self.manager.get_storage_statistics(CacheLevel(level)) for level in range(2)]
        saved = [values(stats) for stats in snapshots]
        for level, stats in enumerate(snapshots):
            self.assertTrue(stats)
            mapping = self.manager.get_life_cycle_pool_group_indices(CacheLevel(level))
            # Both life cycles have the same slot sizes and share one pool group at each level.
            self.assertEqual(mapping, [0, 0])
            self.assertTrue(all(0 <= pool_group < len(stats) for pool_group in mapping))
            mapping[0] = -1
            self.assertNotEqual(
                self.manager.get_life_cycle_pool_group_indices(CacheLevel(level))[0], -1
            )
            for item in stats:
                self.assertEqual(item.total, item.available + item.unavailable)
                self.assertEqual(item.available, item.free + item.evictable)
                sizes = item.slot_sizes
                sizes[0] = 0
            self.assertEqual(
                values(self.manager.get_storage_statistics(CacheLevel(level))), saved[level]
            )

        with TemporaryCudaStream([]) as stream:
            cache = self.manager.create_kv_cache()
            try:
                self.assertTrue(cache.resume(cast(CudaStream, stream.handle)))
                cache.resize(self.manager.tokens_per_block)
                current = self.manager.get_storage_statistics()
                self.assertGreater(sum(item.unavailable for item in current), 0)
                self.assertEqual([values(stats) for stats in snapshots], saved)
            finally:
                cache.close()
        stream.take_finish_event().synchronize()
        self.assertEqual(values(self.manager.get_storage_statistics()), saved[0])


class TestNoBatching(TestKVCacheManagerV2):
    def test_side_stream_waits_for_a_busy_recycled_slot(self) -> None:
        """A full pool hands out a slot its previous owner's stream still writes.

        The cache's stream waits for the slot, so a writer outside every stream,
        such as the NIC, can allocate on a side stream and synchronize only it.
        """
        import torch

        tokens_per_block = 4
        self.manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=tokens_per_block,
                cache_tiers=[GpuCacheTierConfig(4 << 20)],
                layers=[AttentionLayerConfig(0, [BufferConfig("key", 4096)])],
            )
        )
        owner_stream = torch.cuda.Stream()
        loader_stream = torch.cuda.Stream()
        execution_stream = torch.cuda.Stream()

        owner = self.manager.create_kv_cache()
        self.assertTrue(owner.resume(owner_stream.cuda_stream))
        capacity = tokens_per_block
        while owner.resize(capacity + tokens_per_block):
            capacity += tokens_per_block
        with torch.cuda.stream(owner_stream):
            torch.cuda._sleep(1 << 31)
            owner_done = torch.cuda.Event()
            owner_done.record(owner_stream)
        owner.close()

        loader = self.manager.create_kv_cache()
        try:
            self.assertTrue(loader.resume(loader_stream.cuda_stream))
            self.assertTrue(loader.resize(tokens_per_block))
            self.assertFalse(owner_done.query(), "the owner's work finished before the check")
            loader_stream.synchronize()
            self.assertTrue(owner_done.query())
            # Adopting the cache orders the new stream behind the old one.
            loader.cuda_stream = execution_stream.cuda_stream
        finally:
            loader.close()
        execution_stream.synchronize()

    class Request(NamedTuple):
        id: int
        kv_cache: _KVCache
        prompt: list[TokenIdExt]
        decode_len: int

    def new_request(
        self, req_id: int, lora_task_id: int | None, prompt_len: int, decode_len: int
    ) -> Request:
        prompt = [self.next_token() for _ in range(prompt_len)]
        reuse_scope = ReuseScope(lora_id=lora_task_id)
        return self.Request(
            req_id, self.manager.create_kv_cache(reuse_scope, prompt), prompt, decode_len
        )

    def run_request(
        self, req: Request, interval: int, refcheck: bool, delay_commit: bool = False
    ) -> float:
        req_id, kv_cache, prompt, decode_len = req
        assert kv_cache.status == _KVCache.Status.ACTIVE
        stream = kv_cache.cuda_stream
        tic = time.perf_counter()
        # prefill
        num_reused = kv_cache.num_committed_tokens
        if not kv_cache.resize(round_up(len(prompt), interval)):
            raise OutOfPagesError("Not enough pages in GPU memory")
        capacity = kv_cache.capacity
        history = prompt[:num_reused]
        input = prompt[num_reused:]
        if refcheck:
            self.engine.execute([Step(kv_cache, input, history)], stream)
        if input:
            kv_cache.commit(input)
            history.extend(input)
        # decode
        for _ in range(decode_len):
            required_capacity = len(history) + 1
            if required_capacity > capacity:
                if not delay_commit:
                    kv_cache.commit(history[kv_cache.history_length :])
                if not kv_cache.resize(round_up(required_capacity, interval)):
                    raise OutOfPagesError("Not enough pages in GPU memory")
                capacity = kv_cache.capacity
            input_token = self.next_token()
            if refcheck:
                self.engine.execute([Step(kv_cache, [input_token], history)], stream)
            history.append(input_token)
        kv_cache.commit(history[kv_cache.history_length :])
        # last check
        if refcheck:
            self.engine.execute([Step(kv_cache, [], history)], stream)
        toc = time.perf_counter()
        time_taken = toc - tic
        # print(f"Time taken: {time_taken} seconds")
        return time_taken

    def _run_cold_page_codec_round_trip(
        self,
        expected_num_pages: int,
        expected_cold_pages: dict[LayerGroupId, tuple[int, int, int]],
    ) -> None:
        """Force one request through cold storage, then validate its promoted KV."""
        requests: list[TestNoBatching.Request] = []
        try:
            first = self.new_request(0, None, 3 * self.cfg.tokens_per_block, 0)
            requests.append(first)
            with TemporaryCudaStream([]) as stream_holder:
                stream = cast(CudaStream, stream_holder.handle)
                self.assertTrue(first.kv_cache.resume(stream))
                self.run_request(first, self.cfg.tokens_per_block, True)
            stream_holder.take_finish_event().synchronize()
            self.assertEqual(
                _introspection.active_page_stats(first.kv_cache)[0],
                [expected_num_pages, 0],
            )
            first.kv_cache.suspend()
            self.manager.get_and_reset_iteration_stats()

            second = self.new_request(1, None, 3 * self.cfg.tokens_per_block, 0)
            requests.append(second)
            with TemporaryCudaStream([]) as stream_holder:
                stream = cast(CudaStream, stream_holder.handle)
                self.assertTrue(second.kv_cache.resume(stream))
                self.run_request(second, self.cfg.tokens_per_block, True)
            stream_holder.take_finish_event().synchronize()

            # Hot storage has exactly one request worth of slots, so every page of the
            # suspended request must now use the padded cold representation.
            self.assertEqual(
                _introspection.active_page_stats(first.kv_cache)[0],
                [0, expected_num_pages],
            )
            offload_stats = self.manager.get_and_reset_iteration_stats()
            for life_cycle_id, (offload_pages, _, page_bytes) in expected_cold_pages.items():
                self.assertEqual(offload_stats[life_cycle_id].iter_offload_blocks, offload_pages)
                self.assertEqual(
                    offload_stats[life_cycle_id].iter_offload_bytes,
                    offload_pages * page_bytes,
                )

            second.kv_cache.close()
            with TemporaryCudaStream([]) as stream_holder:
                stream = cast(CudaStream, stream_holder.handle)
                self.assertTrue(first.kv_cache.resume(stream))
                self.run_request(first, self.cfg.tokens_per_block, True)
            stream_holder.take_finish_event().synchronize()
            self.assertEqual(
                _introspection.active_page_stats(first.kv_cache)[0],
                [expected_num_pages, 0],
            )
            onboard_stats = self.manager.get_and_reset_iteration_stats()
            for life_cycle_id, (_, onboard_pages, page_bytes) in expected_cold_pages.items():
                self.assertEqual(onboard_stats[life_cycle_id].iter_onboard_blocks, onboard_pages)
                self.assertEqual(
                    onboard_stats[life_cycle_id].iter_onboard_bytes,
                    onboard_pages * page_bytes,
                )
        finally:
            for request in requests:
                if request.kv_cache.status != _KVCache.Status.CLOSED:
                    request.kv_cache.close()
            if hasattr(self, "manager"):
                self.manager.clear_reusable_blocks()

    def test_cold_codec_merges_lifecycles_from_different_hot_pool_groups(self) -> None:
        """Padding merges full attention with one of two differently-sized SWA LCs."""
        unit = 1 << 20
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[
                GpuCacheTierConfig(quota=24 * unit),
                GpuCacheTierConfig(quota=64 * unit),
            ],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role=Role.KEY, size=4 * unit)],
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[BufferConfig(role=Role.KEY, size=2 * unit)],
                    sliding_window_size=4,
                    num_sink_tokens=0,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(2),
                    buffers=[BufferConfig(role=Role.KEY, size=2 * unit)],
                    sliding_window_size=8,
                    num_sink_tokens=0,
                ),
            ],
            initial_pool_ratio=[1 / 3, 1 / 3, 1 / 3],
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=12, history_length=0)])],
            max_util_for_resume=1.0,
        )
        self.engine = FakeEngine(self.cfg)
        codec = _introspection.create_test_padding_cold_page_codec(
            {0: 4 * unit, 1: 4 * unit, 2: 2 * unit}
        )
        self.manager = KVCacheManager(self.cfg, cold_page_codec=codec)

        full_lc, short_swa_lc, long_swa_lc = [
            self.manager.get_layer_group_id(LayerId(layer_id)) for layer_id in range(3)
        ]
        hot_mapping = self.manager.get_life_cycle_pool_group_indices(GPU_LEVEL)
        cold_mapping = self.manager.get_life_cycle_pool_group_indices(CacheLevel(1))
        hot_groups = [hot_mapping[lc_id] for lc_id in (full_lc, short_swa_lc, long_swa_lc)]
        cold_groups = [cold_mapping[lc_id] for lc_id in (full_lc, short_swa_lc, long_swa_lc)]
        self.assertNotEqual(hot_groups[0], hot_groups[1])
        self.assertEqual(hot_groups[1], hot_groups[2])
        self.assertEqual(cold_groups[0], cold_groups[1])
        self.assertNotEqual(cold_groups[1], cold_groups[2])

        hot_stats = self.manager.get_storage_statistics(GPU_LEVEL)
        self.assertEqual(hot_stats[hot_groups[0]].total, 3)
        self.assertEqual(hot_stats[hot_groups[1]].total, 6)
        self._run_cold_page_codec_round_trip(
            expected_num_pages=6,
            expected_cold_pages={
                full_lc: (3, 3, 4 * unit),
                short_swa_lc: (3, 1, 4 * unit),
                long_swa_lc: (3, 2, 2 * unit),
            },
        )

    def test_cold_codec_splits_lifecycles_from_one_hot_pool_group(self) -> None:
        """Padding one SWA lifecycle splits a shared hot pool group in cold storage."""
        unit = 1 << 20
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[
                GpuCacheTierConfig(quota=12 * unit),
                GpuCacheTierConfig(quota=32 * unit),
            ],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role=Role.KEY, size=2 * unit)],
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[BufferConfig(role=Role.KEY, size=2 * unit)],
                    sliding_window_size=8,
                    num_sink_tokens=0,
                ),
            ],
            initial_pool_ratio=[0.5, 0.5],
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=12, history_length=0)])],
            max_util_for_resume=1.0,
        )
        self.engine = FakeEngine(self.cfg)
        codec = _introspection.create_test_padding_cold_page_codec({0: 2 * unit, 1: 4 * unit})
        self.manager = KVCacheManager(self.cfg, cold_page_codec=codec)

        full_lc, swa_lc = [
            self.manager.get_layer_group_id(LayerId(layer_id)) for layer_id in range(2)
        ]
        hot_mapping = self.manager.get_life_cycle_pool_group_indices(GPU_LEVEL)
        cold_mapping = self.manager.get_life_cycle_pool_group_indices(CacheLevel(1))
        hot_groups = [hot_mapping[lc_id] for lc_id in (full_lc, swa_lc)]
        cold_groups = [cold_mapping[lc_id] for lc_id in (full_lc, swa_lc)]
        self.assertEqual(hot_groups[0], hot_groups[1])
        self.assertNotEqual(cold_groups[0], cold_groups[1])

        hot_stats = self.manager.get_storage_statistics(GPU_LEVEL)
        self.assertEqual(hot_stats[hot_groups[0]].total, 6)
        self._run_cold_page_codec_round_trip(
            expected_num_pages=5,
            expected_cold_pages={full_lc: (3, 3, 2 * unit), swa_lc: (3, 2, 4 * unit)},
        )

    def run_naive(
        self,
        seq_len: int,
        interval: int = 1,
        refcheck: bool = True,
        use_external_page_index_buf: bool = False,
        delay_commit: bool = False,
    ) -> float:
        prompt_len = 1
        decode_len = seq_len - prompt_len

        req_id = 0
        lora_task_id = None
        req0 = self.new_request(req_id, lora_task_id, prompt_len, decode_len)
        if use_external_page_index_buf:
            max_num_blocks = div_up(seq_len, self.cfg.tokens_per_block)
            num_layer_groups = len(self.manager.layer_grouping)
            base_page_indices = [
                array.array("i", [-1]) * max_num_blocks for _ in range(num_layer_groups)
            ]
            for id in range(num_layer_groups):
                req0.kv_cache.set_base_page_index_buf(
                    DEFAULT_BEAM_INDEX, LayerGroupId(id), memoryview(base_page_indices[id])
                )
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv_cache = req0.kv_cache
            success = kv_cache.resume(stream)
            assert success
            time_taken = self.run_request(req0, interval, refcheck, delay_commit)

        s.take_finish_event().synchronize()
        kv_cache.close()
        self.manager.clear_reusable_blocks()
        return time_taken

    @parameterized.expand([(False,), (True,)])
    def test_shrink_capacity(self, use_external_page_index_buf: bool) -> None:
        self.prepare(32 << 20, 32 << 20, 1 << 30, 36, 128, 1, kv_buf_size=32768)
        seq_len = 32 * 10
        req0 = self.new_request(0, None, 32, seq_len - 32)
        if use_external_page_index_buf:
            max_num_blocks = div_up(seq_len, self.cfg.tokens_per_block)
            num_layer_groups = len(self.manager.layer_grouping)
            base_page_indices = [
                array.array("i", [-1]) * max_num_blocks for _ in range(num_layer_groups)
            ]
            for id in range(num_layer_groups):
                req0.kv_cache.set_base_page_index_buf(
                    DEFAULT_BEAM_INDEX, LayerGroupId(id), memoryview(base_page_indices[id])
                )
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv_cache = req0.kv_cache
            success = kv_cache.resume(stream)
            assert success
            success = kv_cache.resize(seq_len)
            assert success
            for capacity in range(seq_len, len(req0.prompt), -1):
                success = kv_cache.resize(capacity)
                assert success
        s.take_finish_event()
        kv_cache.close()

    def test_small_quota(self) -> None:
        self.prepare(5619712, 0, 0, 8, None, 0)
        assert self.manager.get_quota(cast(CacheLevel, GPU_LEVEL)) >= 5619712

    # @assert_no_ref_cycle
    def test_sol_mem_utilization(self) -> None:
        self.prepare(32 << 20, 32 << 20, 1 << 30, 36, 128, 1, kv_buf_size=32768)
        # if we have n blocks, we need 8192*2*18*(1+5+n) bytes of memory. For the (1+5+n), 1 is for sink
        # blocks, 5 is for SWA (window=128), n is for full attention.
        max_seq_len = 32 * 22  # 23 blocks will require more than 32MB memory
        seq_len = max_seq_len

        # create a request and suspend it. It shall not consume any GPU memory after suspend.
        req0 = self.new_request(0, None, 256, seq_len - 256)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = req0.kv_cache.resume(stream)
            assert success
            self.run_request(req0, 32, False)
        s.take_finish_event()
        req0.kv_cache.suspend()

        # run another request that will take all the GPU memory
        req1 = self.new_request(0, None, 256, seq_len - 256)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = req1.kv_cache.resume(stream)
            assert success
            self.run_request(req1, 1, True)
        s.take_finish_event()

        req1.kv_cache.close()
        req0.kv_cache.close()

        # run another longer request and expect OutOfPagesError
        # This also tests eviction to disk.
        self.assertRaises(OutOfPagesError, lambda: self.run_naive(seq_len + 1, 1, False))

    def test_resume_rejects_if_any_pool_group_exceeds_threshold(self) -> None:
        cfg = KVCacheManagerConfig(
            tokens_per_block=32,
            cache_tiers=[GpuCacheTierConfig(quota=4 << 20)],
            max_util_for_resume=0.9,
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role=Role.KEY, size=(1 << 20) + 1)],
                    sliding_window_size=32,
                    num_sink_tokens=0,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[BufferConfig(role=Role.KEY, size=1024)],
                    sliding_window_size=None,
                ),
            ],
            typical_step=BatchDesc(kv_caches=[KVCacheDesc(capacity=32, history_length=0)]),
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=32, history_length=0)])],
        )
        self.manager = KVCacheManager(cfg)

        def stat_slot_sizes(stat) -> list[int]:
            if hasattr(stat, "slot_sizes"):
                return stat.slot_sizes
            return stat.slot_size

        def overall_utilization() -> float:
            numerator = 0
            denominator = 0
            for stat in self.manager.get_storage_statistics():
                slot_size = sum(stat_slot_sizes(stat))
                numerator += slot_size * stat.unavailable
                denominator += slot_size * stat.total
            return numerator / denominator

        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prior_caches: list[_KVCache] = []
        try:
            # The worst-case SWA slot reservation means a single sequence cannot
            # push a pool group past max_util_for_resume, so resume sequences until
            # the big-slot SWA pool group crosses the threshold.
            for _ in range(64):
                if max(_introspection.storage_utilization(self.manager, GPU_LEVEL)) > (
                    cfg.max_util_for_resume
                ):
                    break
                kv_cache = self.manager.create_kv_cache()
                if not kv_cache.resume(stream):
                    kv_cache.close()
                    break
                self.assertTrue(kv_cache.resize(cfg.tokens_per_block))
                prior_caches.append(kv_cache)

            utilizations = _introspection.storage_utilization(self.manager, GPU_LEVEL)
            self.assertGreater(max(utilizations), cfg.max_util_for_resume)
            self.assertLess(overall_utilization(), cfg.max_util_for_resume)

            # One pool group is now over the limit, so a further resume is rejected.
            rejected_cache = self.manager.create_kv_cache()
            prior_caches.append(rejected_cache)
            self.assertFalse(rejected_cache.resume(stream))
            self.assertEqual(rejected_cache.status, _KVCache.Status.SUSPENDED)
        finally:
            for kv_cache in prior_caches:
                if kv_cache.status != _KVCache.Status.CLOSED:
                    kv_cache.close()

    @parameterized.expand([(1,), (2,), (4,)])
    # @assert_no_ref_cycle
    def test_cache_reuse(self, num_reusable_requests: int) -> None:
        self.prepare(32 << 20, 32 << 20, 1 << 30, 36, 128, 1, kv_buf_size=32768)
        # if we have n blocks, we need 8192*2*18*(1+5+n) bytes of memory. For the (1+5+n), 1 is for sink
        # blocks, 5 is for SWA (window=128), n is for full attention.
        max_seq_len = 32 * 22  # 23 blocks will require more than 32MB memory
        seq_len = max_seq_len

        req_id_gen = itertools.count()
        reusable_requests = []
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            for _ in range(num_reusable_requests):
                req = self.new_request(next(req_id_gen), None, 256, seq_len - 256)
                reusable_requests.append(req)
                success = req.kv_cache.resume(stream)
                assert success
                self.run_request(req, 32, True)
                req.kv_cache.close()
        s.take_finish_event()

        self.assertTrue(_introspection.all_tree_pages_droppable(self.manager))

        req0 = reusable_requests[0]
        prompt1 = req0.kv_cache.committed_tokens[: (seq_len // 2 - 7)]
        # request id must be same as req0 because we wrote it into the kv cache.
        req1 = self.Request(
            next(req_id_gen),
            self.manager.create_kv_cache(None, prompt1),
            prompt1,
            seq_len - len(prompt1),
        )
        assert req1.kv_cache.num_committed_tokens == len(prompt1)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = req1.kv_cache.resume(stream)
            assert success
            self.run_request(req1, 32, True)
        s.take_finish_event()
        req1.kv_cache.close()

        self.manager.clear_reusable_blocks()

    def test_commit_min_snapshot_reuses_swa_post_commit_prefix(self) -> None:
        tokens_per_block = 32
        window_size = 64
        prompt = [TokenId(i) for i in range(tokens_per_block * 4)]
        cfg = KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=[GpuCacheTierConfig(quota=16 << 20)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=8192),
                        BufferConfig(role=Role.VALUE, size=8192),
                    ],
                    sliding_window_size=window_size,
                )
            ],
            commit_min_snapshot=True,
        )
        self.manager = KVCacheManager(cfg)

        with TemporaryCudaStream([]) as stream_holder:
            stream = cast(CudaStream, stream_holder.handle)
            kv1 = self.manager.create_kv_cache()
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(len(prompt), len(prompt)))
            kv1.commit(prompt)
            kv1.close()
        stream_holder.take_finish_event().synchronize()

        swa_lc_id = _introspection.swa_life_cycle_ids(self.manager)[0]
        num_tokens, pages = _introspection.reuse_match_pages(
            self.manager, ReuseScope(), prompt, swa_lc_id
        )
        self.assertEqual(num_tokens, len(prompt))
        self.assertEqual(len(pages), 4)
        # The committed snapshot is reusable at the post-commit token count, but
        # old SWA blocks outside that window should not keep reusable pages.
        self.assertIsNone(pages[0])
        self.assertIsNone(pages[1])
        self.assertIsNotNone(pages[2])
        self.assertIsNotNone(pages[3])
        self.assertEqual(
            self.manager.probe_reuse(input_tokens=prompt[: tokens_per_block * 3]),
            0,
        )

        kv2 = self.manager.create_kv_cache(input_tokens=prompt)
        self.assertEqual(kv2.num_committed_tokens, len(prompt))
        kv2.close()

    def test_reuse_match_backoff_trims_the_tail_of_every_match(self) -> None:
        """A pool holding state that reads D tokens ahead cannot use the last D.

        Set for a one-model speculative decoding draft pool, whose KV at position
        i is a function of tokens [0, i + D]: a match of m tokens only describes
        the first m - D positions. Applied inside match(), so probe_reuse and the
        claim agree and neither needs a second tree walk. Commit is untouched --
        the blocks are stored in full; only what a later match may use shrinks.
        """
        tokens_per_block = 8
        backoff = 3
        prompt = [TokenId(i) for i in range(tokens_per_block * 4)]
        self.manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=tokens_per_block,
                cache_tiers=[GpuCacheTierConfig(quota=16 << 20)],
                layers=[
                    AttentionLayerConfig(
                        layer_id=LayerId(0),
                        buffers=[
                            BufferConfig(role=Role.KEY, size=1024),
                            BufferConfig(role=Role.VALUE, size=1024),
                        ],
                    )
                ],
                reuse_match_backoff=backoff,
            )
        )

        with TemporaryCudaStream([]) as stream_holder:
            stream = cast(CudaStream, stream_holder.handle)
            kv = self.manager.create_kv_cache()
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(len(prompt), len(prompt)))
            kv.commit(prompt)
            self.assertEqual(kv.num_committed_tokens, len(prompt))
            kv.close()
        stream_holder.take_finish_event().synchronize()

        # Every other test in this file runs with the default backoff of 0 and
        # matches the full prompt, so the delta here is the backoff alone. The
        # trim crosses a block boundary rather than stopping at one.
        self.assertEqual(self.manager.probe_reuse(None, prompt), len(prompt) - backoff)

        # A match shorter than the backoff collapses to nothing, not negative.
        self.assertEqual(self.manager.probe_reuse(None, prompt[: backoff - 1]), 0)

        # The claim agrees with the probe: same trim, no second tree walk.
        kv2 = self.manager.create_kv_cache(input_tokens=prompt)
        self.assertEqual(kv2.num_committed_tokens, len(prompt) - backoff)
        kv2.close()

    @parameterized.expand([("backoff_0", 0), ("backoff_1", 1)])
    def test_planned_drop_handle(self, _case: str, reuse_match_backoff: int) -> None:
        window_size = 8
        self.cfg = create_config(8, 16 << 20, 0, 0, 2, window_size, 0)
        self.cfg.reuse_match_backoff = reuse_match_backoff
        self.manager = KVCacheManager(self.cfg)
        long_tokens = [self.next_token() for _ in range(24)]
        short_tokens = long_tokens[:8]

        def plan_drop(tokens: list[TokenIdExt]) -> PlannedDropHandle:
            kv_cache = self.manager.create_kv_cache(None, tokens)
            with TemporaryCudaStream([]) as stream_holder:
                stream = cast(CudaStream, stream_holder.handle)
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(len(tokens)))
                uncommitted = tokens[kv_cache.num_committed_tokens :]
                if uncommitted:
                    kv_cache.commit(uncommitted)
                kv_cache.stop_committing()
                drop_handle = kv_cache.plan_committed_block_drop()
                self.assertIsNotNone(drop_handle)
                self.assertIsInstance(drop_handle, PlannedDropHandle)
            _ = stream_holder.take_finish_event()
            kv_cache.close()
            assert drop_handle is not None
            return drop_handle

        long_handle = plan_drop(long_tokens)
        short_handle = plan_drop(short_tokens)
        self.assertEqual(
            self.manager.probe_reuse(None, short_tokens),
            len(short_tokens) - reuse_match_backoff,
        )

        short_handle.drop()
        self.assertEqual(self.manager.probe_reuse(None, short_tokens), 0)
        self.assertEqual(
            self.manager.probe_reuse(None, long_tokens),
            len(long_tokens) - reuse_match_backoff,
        )

        long_handle.drop()
        # The SWA window is dropped, while older full-attention blocks remain reusable.
        self.assertEqual(
            self.manager.probe_reuse(None, long_tokens), len(long_tokens) - window_size
        )
        with self.assertRaisesRegex(RuntimeError, "already been dropped"):
            long_handle.drop()

    def test_int32_ndarray_ingest_matches_list(self) -> None:
        """Zero-copy int32-ndarray ingest must hash identically to the list path.

        The int32-ndarray path must agree with the per-element list path across
        create_kv_cache / commit / probe_reuse.

        A digest-free int32 token is bit-identical to a normal 4-byte TokenIdExt,
        so the C++ binding reinterprets a contiguous int32 buffer to TokenIdExt*
        with no copy. If that reinterpret disagreed with the list path by even one
        bit, blocks committed via the ndarray path would not be found by a list
        probe (and vice versa), so the equalities below would fail.
        """
        import numpy as np

        tokens_per_block = 8
        self.prepare(16 << 20, 0, 0, 2, None, 0, tokens_per_block=tokens_per_block)
        # Pure-int prompt (no randbytes/digest tokens) so the int32 fast path applies.
        prompt = [TokenId(i) for i in range(tokens_per_block * 3)]
        prompt_np = np.asarray(prompt, dtype=np.int32)
        assert prompt_np.dtype == np.int32 and prompt_np.flags["C_CONTIGUOUS"]

        def commit_prompt(tokens: "Sequence[TokenIdExt] | np.ndarray") -> None:
            kv_cache = self.manager.create_kv_cache(None, tokens)
            with TemporaryCudaStream([]) as stream_holder:
                stream = cast(CudaStream, stream_holder.handle)
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(len(prompt)))
                committed = kv_cache.num_committed_tokens
                if committed < len(prompt):
                    kv_cache.commit(tokens[committed:])
                kv_cache.stop_committing()
            _ = stream_holder.take_finish_event()
            kv_cache.close()

        # Commit via the int32-ndarray fast path (both create_kv_cache and commit).
        commit_prompt(prompt_np)

        # Probing with a list and with an int32 ndarray must both fully match —
        # proving the ndarray commit hashes like the list path, and the ndarray
        # probe hashes like the list probe.
        self.assertEqual(self.manager.probe_reuse(None, prompt), len(prompt))
        self.assertEqual(self.manager.probe_reuse(None, prompt_np), len(prompt))

        # Reverse direction: commit via the list path in a fresh tree, probe via
        # the int32-ndarray path → full match.
        self.manager.clear_reusable_blocks()
        self.assertEqual(self.manager.probe_reuse(None, prompt_np), 0)
        commit_prompt(prompt)
        self.assertEqual(self.manager.probe_reuse(None, prompt_np), len(prompt))

    def test_reuse_scope_isolates_reuse(self) -> None:
        self.prepare(16 << 20, 0, 0, 2, None, 0, tokens_per_block=8)
        tokens = [TokenId(i) for i in range(64)]
        capacity = 128
        default_scope = ReuseScope()
        scoped = ReuseScope(lora_id=7, salt="11")

        def commit_for(reuse_scope: ReuseScope | None) -> None:
            kv_cache = self.manager.create_kv_cache(reuse_scope, tokens[:-1])
            self.assertEqual(kv_cache.reuse_scope, reuse_scope or default_scope)
            with TemporaryCudaStream([]) as stream_holder:
                stream = cast(CudaStream, stream_holder.handle)
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(capacity))
                uncommitted = tokens[kv_cache.num_committed_tokens :]
                if uncommitted:
                    kv_cache.commit(uncommitted)
                kv_cache.stop_committing()
            stream_holder.take_finish_event()
            kv_cache.close()

        def num_reused(reuse_scope: ReuseScope | None) -> int:
            probed = self.manager.probe_reuse(reuse_scope, tokens[:-1])
            kv_cache = self.manager.create_kv_cache(reuse_scope, tokens[:-1])
            self.assertEqual(kv_cache.reuse_scope, reuse_scope or default_scope)
            ret = kv_cache.num_committed_tokens
            kv_cache.close()
            self.assertEqual(probed, ret)
            return ret

        commit_for(scoped)
        self.assertGreater(num_reused(scoped), 0)
        self.assertEqual(num_reused(ReuseScope(lora_id=7, salt="12")), 0)
        self.assertEqual(num_reused(ReuseScope(lora_id=8, salt="11")), 0)
        self.assertEqual(num_reused(ReuseScope(lora_id=7)), 0)
        self.assertEqual(num_reused(ReuseScope(salt="11")), 0)
        self.assertEqual(num_reused(None), 0)

        commit_for(None)
        self.assertGreater(num_reused(None), 0)
        self.assertGreater(num_reused(default_scope), 0)

    def test_create_kv_cache_accepts_sequence_input_tokens(self) -> None:
        self.prepare(8 << 20, 0, 0, 2, None, 0, tokens_per_block=4, kv_buf_size=1024)
        prompt = [self.next_token() for _ in range(8)]

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv_cache = self.manager.create_kv_cache()
            try:
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(len(prompt), len(prompt)))
                kv_cache.commit(prompt)
                kv_cache.stop_committing()
            finally:
                if kv_cache.status != _KVCache.Status.CLOSED:
                    kv_cache.close()
        s.take_finish_event()

        kv_cache = self.manager.create_kv_cache(input_tokens=tuple(prompt))
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            try:
                self.assertTrue(kv_cache.resume(stream))
                self.assertEqual(kv_cache.num_committed_tokens, len(prompt))
            finally:
                if kv_cache.status != _KVCache.Status.CLOSED:
                    kv_cache.close()
        s.take_finish_event()

    def test_create_kv_cache_custom_priority_callback_gets_lifecycle(self) -> None:
        self.prepare(8 << 20, 0, 0, 2, 128, 0, tokens_per_block=4, kv_buf_size=1024)
        seen_life_cycles = []

        def custom_priority_callback(_ordinal, life_cycle):
            seen_life_cycles.append(life_cycle)
            self.assertNotIsInstance(life_cycle, int)
            self.assertTrue(hasattr(life_cycle, "get_stale_range"))
            return 42

        kv_cache = self.manager.create_kv_cache(custom_priority_callback=custom_priority_callback)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            try:
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(4, 4))
            finally:
                if kv_cache.status != _KVCache.Status.CLOSED:
                    kv_cache.close()
        s.take_finish_event()

        self.assertTrue(seen_life_cycles)

    def test_cached_cuda_event_constructor_and_null(self) -> None:
        cached_cuda_event = get_cached_cuda_event_type()
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            event = cached_cuda_event(stream)
            self.assertFalse(event.is_closed())
            event.wait_in_stream(stream)
            event.synchronize()
            self.assertTrue(event.is_closed())

            null_event = cached_cuda_event.NULL
            self.assertTrue(null_event.is_closed())
            self.assertTrue(null_event.query_complete())
            null_event.synchronize()
            null_event.wait_in_stream(stream)
        s.take_finish_event()

    def test_base_page_index_external_buffer_validation(self) -> None:
        self.prepare(8 << 20, 0, 0, 2, None, 0, tokens_per_block=4, kv_buf_size=1024)
        kv_cache = self.manager.create_kv_cache()
        try:
            with TemporaryCudaStream([]) as s:
                stream = cast(CudaStream, s.handle)
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(8))
                num_blocks = kv_cache.num_blocks

                undersized = array.array("i", [BAD_PAGE_INDEX]) * (num_blocks - 1)
                with self.assertRaises((AssertionError, ValueError)):
                    kv_cache.set_base_page_index_buf(
                        DEFAULT_BEAM_INDEX, LayerGroupId(0), memoryview(undersized)
                    )

                oversized = array.array("i", [123]) * (num_blocks + 2)
                kv_cache.set_base_page_index_buf(
                    DEFAULT_BEAM_INDEX, LayerGroupId(0), memoryview(oversized)
                )
                self.assertEqual(list(oversized[num_blocks:]), [BAD_PAGE_INDEX, BAD_PAGE_INDEX])
                kv_cache.close()
            s.take_finish_event()
        finally:
            if kv_cache.status != _KVCache.Status.CLOSED:
                kv_cache.close()

    def test_buffer_id_tuple_hash_protocol(self) -> None:
        buffer_id = BufferId(LayerId(1), Role.KEY)
        same_buffer_id = BufferId(LayerId(1), Role.KEY)
        as_tuple = (LayerId(1), Role.KEY)

        self.assertEqual(tuple(buffer_id), as_tuple)
        self.assertEqual(buffer_id[0], as_tuple[0])
        self.assertEqual(buffer_id[-1], as_tuple[1])
        self.assertEqual(len(buffer_id), 2)
        self.assertEqual(buffer_id, as_tuple)
        self.assertEqual(as_tuple, buffer_id)
        self.assertEqual(buffer_id, same_buffer_id)
        self.assertEqual(hash(buffer_id), hash(as_tuple))
        self.assertEqual({buffer_id: 7}[same_buffer_id], 7)
        self.assertEqual({buffer_id: 7}[as_tuple], 7)
        with self.assertRaises(AttributeError):
            buffer_id.layer_id = LayerId(2)

    def test_shrink_capacity_truncates_base_page_indices(self) -> None:
        self.prepare(8 << 20, 0, 0, 2, None, 0, tokens_per_block=4, kv_buf_size=1024)
        kv_cache = self.manager.create_kv_cache()
        layer_group = self.manager.get_layer_group_id(LayerId(0))

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            try:
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(8, 0))
                self.assertEqual(kv_cache.num_blocks, 2)
                self.assertEqual(len(kv_cache.get_base_page_indices(layer_group)), 2)

                self.assertTrue(kv_cache.resize(4, 0))
                self.assertEqual(kv_cache.num_blocks, 1)
                self.assertEqual(len(kv_cache.get_base_page_indices(layer_group)), 1)
            finally:
                if kv_cache.status != _KVCache.Status.CLOSED:
                    kv_cache.close()
        s.take_finish_event()

    @parameterized.expand(list(itertools.product([False, True], repeat=3)))
    # @assert_no_ref_cycle
    def test_naive(
        self,
        use_external_page_index_buf: bool,
        use_block_quant: bool,
        delay_commit: bool,
    ) -> None:
        self.prepare(
            256 << 20,
            256 << 20,
            1 << 30,
            36,
            128,
            48,
            block_quant_buf_size=(1024 if use_block_quant else None),
        )
        self.run_naive(512, 1, True, use_external_page_index_buf, delay_commit=delay_commit)

    @parameterized.expand([(2**i, False) for i in range(12)])
    # @parameterized.expand([(32, True)])
    # @assert_no_ref_cycle
    def test_naive_perf(self, interval, profile: bool) -> None:
        if not PRINT_TIME:
            self.skipTest("Skipping perf test")
        self.prepare(256 << 20, 256 << 20, 1 << 30, 36, 128, 48)
        seq_len = 10240
        self.run_naive(seq_len, interval, False)  # warm up for numba jit
        profiler = None
        if profile:
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
        time_taken = [
            self.run_naive(seq_len, interval, False) for _ in range(11 if profiler is None else 1)
        ]
        median_time_taken = median(time_taken)
        if PRINT_TIME:
            print(
                f"Throughput: {round(seq_len / median_time_taken)} tokens/sec for interval {interval}"
            )
        if profiler is not None:
            profiler.disable()
            profiler.print_stats(sort="cumtime")
            profiler.dump_stats("profiler.prof")


class TestLivingKvCacheGuard(TestKVCacheManagerV2):
    """Guard against clearing/freeing the reuse state while KV caches are still open.

    `clear_reusable_blocks()` detaches the whole radix tree and `shutdown()` frees the
    storage the pages live in. A request that is still open keeps committing into the
    detached subtree, which segfaults, so both entry points must reject the call instead.
    """

    # TLLM_CHECK_WITH_INFO throws a TllmException (a std::runtime_error), which nanobind
    # surfaces as RuntimeError.
    GuardError = RuntimeError

    def _seed_reusable_prompt(self) -> list[TokenIdExt]:
        """Commit and close a sequence so the radix tree actually holds reusable blocks.

        Without this the tree is empty, and a regression that cleared it *before* raising
        would still pass — there would be nothing left to observe.
        """
        prompt = [self.next_token() for _ in range(64)]
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        seed = self.manager.create_kv_cache()
        seed.resume(stream)
        seed.capacity = 32
        seed.commit(prompt[:32])
        seed.capacity = 64
        seed.commit(prompt[32:])
        seed.stop_committing()
        seed.close()
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 64)
        return prompt

    def _open_cache(self) -> _KVCache:
        return self.manager.create_kv_cache(
            ReuseScope(lora_id=None), [self.next_token() for _ in range(64)]
        )

    def test_clear_reusable_blocks_rejects_open_kv_cache(self) -> None:
        self.prepare(32 << 20, 32 << 20, 1 << 30, 4, 128, 1)
        prompt = self._seed_reusable_prompt()
        kv_cache = self._open_cache()
        try:
            with self.assertRaises(self.GuardError) as ctx:
                self.manager.clear_reusable_blocks()
            # The message must name the API the caller actually invoked, and report how
            # many sequences are still open.
            self.assertIn("clear_reusable_blocks()", str(ctx.exception))
            self.assertIn("1 KV cache(s) still open", str(ctx.exception))
            # The rejected call must be a no-op: the check runs before the tree is
            # touched, so every block is still reusable.
            self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 64)
        finally:
            kv_cache.close()

        # ...and once permitted it really does clear, which is what stops the assertion
        # above from passing vacuously.
        self.manager.clear_reusable_blocks()
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 0)

    def test_shutdown_rejects_open_kv_cache(self) -> None:
        self.prepare(32 << 20, 32 << 20, 1 << 30, 4, 128, 1)
        prompt = self._seed_reusable_prompt()
        kv_cache = self._open_cache()
        try:
            with self.assertRaises(self.GuardError) as ctx:
                self.manager.shutdown()
            self.assertIn("shutdown()", str(ctx.exception))
            self.assertIn("1 KV cache(s) still open", str(ctx.exception))
            # shutdown() frees the storage the pages live in, so a rejected call must
            # leave both the reuse state and the pool intact: the blocks are still
            # reusable, and the manager can still hand out and resume a new sequence.
            self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 64)
            stream_holder = CachedCudaStream()
            probe = self.manager.create_kv_cache()
            probe.resume(cast(CudaStream, stream_holder.handle))
            probe.close()
        finally:
            kv_cache.close()

        self.manager.shutdown()
        del self.manager

    def test_guard_counts_only_open_caches(self) -> None:
        """The guard counts sequences still open, not objects still referenced."""
        self.prepare(32 << 20, 32 << 20, 1 << 30, 4, 128, 1)
        caches = [
            self.manager.create_kv_cache(
                ReuseScope(lora_id=None), [self.next_token() for _ in range(64)]
            )
            for _ in range(3)
        ]
        try:
            with self.assertRaises(self.GuardError) as ctx:
                self.manager.clear_reusable_blocks()
            self.assertIn("3 KV cache(s) still open", str(ctx.exception))

            caches[0].close()
            with self.assertRaises(self.GuardError) as ctx:
                self.manager.clear_reusable_blocks()
            self.assertIn("2 KV cache(s) still open", str(ctx.exception))
        finally:
            # Close every cache, not just caches[1:]: if an assertion above fails before
            # caches[0] is closed, leaving it open makes tearDown's shutdown() raise and
            # mask the real failure. close() is idempotent, so the double close is fine.
            for kv_cache in caches:
                kv_cache.close()

        # `caches` still holds all three references, so a passing call here proves the
        # guard tracks close() rather than object liveness.
        self.manager.clear_reusable_blocks()


class TestBatching(TestKVCacheManagerV2):
    num_requests: int
    avg_length: int
    past_sequences: list[list[TokenIdExt]]
    seq_len_dict: dict[_KVCache, int]
    batch: list[Step]
    suspended: list[Step]
    num_created: int
    num_finished: int
    req_id_gen: Iterator[int]
    acc_num_prompt_tokens: int
    acc_num_decode_tokens: int
    interval: int
    enable_reuse: bool

    def setUp(self) -> None:
        super().setUp()
        self.past_sequences = list[list[TokenIdExt]]()
        self.seq_len_dict = dict[_KVCache, int]()
        self.batch = list[Step]()
        self.suspended = list[Step]()
        self.num_finished = 0
        self.num_created = 0
        self.req_id_gen = itertools.count()
        self.acc_num_prompt_tokens = 0
        self.acc_num_decode_tokens = 0
        self.enable_reuse = False

    def gen_request(self) -> Step:
        if self.num_created >= self.num_requests:
            raise ValueError("Too many requests created")

        token_id_gen = cast(Iterator[TokenId], self._token_id_gen)

        def gen_length() -> int:
            return random.randint(int(self.avg_length * 0.6), int(self.avg_length * 1.4))

        if self.enable_reuse:
            if len(self.past_sequences) >= 32 and random.random() < 0.2:
                # continued multi-round dialog
                prompt = random.choice(self.past_sequences) + [
                    next(token_id_gen) for _ in range(gen_length())
                ]
            else:
                # new dialog
                if len(self.past_sequences) < 32 or random.random() < 0.5:
                    # completely new prompt
                    prompt = [next(token_id_gen) for _ in range(gen_length())]
                else:
                    # with reused tokens
                    reused = random.choice(self.past_sequences)
                    prompt = reused[: random.randint(0, min(gen_length(), len(reused)))] + [
                        next(token_id_gen) for _ in range(gen_length())
                    ]
        else:
            prompt = [next(token_id_gen) for _ in range(gen_length())]
        decode_len = gen_length()
        lora_task_id = None
        reuse_scope = ReuseScope(lora_id=lora_task_id)
        kv_cache = self.manager.create_kv_cache(
            reuse_scope, prompt[:-1] if self.enable_reuse else None, id=next(self.req_id_gen)
        )
        DBG_PRINT and print(  # type: ignore[arg-type]
            f"created {kv_cache.id} with {kv_cache.num_committed_tokens} tokens reused"
        )
        history = prompt[: kv_cache.num_committed_tokens]
        input = prompt[kv_cache.num_committed_tokens :]
        seq_len = len(prompt) + decode_len
        self.seq_len_dict[kv_cache] = seq_len
        self.num_created += 1
        assert input
        self.acc_num_prompt_tokens += len(prompt)
        self.acc_num_decode_tokens += decode_len
        return Step(kv_cache, input, history)

    def update_batch(self, stream: CudaStream) -> None:
        for s in self.batch:
            assert s.input
            if self.enable_reuse:
                s.kv_cache.commit(s.input)
            else:
                s.kv_cache.history_length += len(s.input)
            s.history.extend(s.input)
            s.input.clear()
        # remove finished requests first
        removed = remove_if(
            self.batch,
            lambda step: len(step.history) >= self.seq_len_dict[step.kv_cache],
        )
        for kv_cache, _, _ in removed:
            seq_len = self.seq_len_dict[kv_cache]
            if seq_len < self.avg_length * 3:
                self.past_sequences.append(kv_cache.committed_tokens[:seq_len])
            kv_cache.close()
            self.seq_len_dict.pop(kv_cache)
            self.num_finished += 1
        # fill input for remaining requests and increase capacity for them
        token_id_gen = cast(Iterator[TokenId], self._token_id_gen)
        for s in self.batch:
            assert not s.input
            length = min(self.interval, self.seq_len_dict[s.kv_cache] - len(s.history))
            s.input.extend(next(token_id_gen) for _ in range(length))
        for i in itertools.count():
            if i >= len(self.batch):
                break
            s = self.batch[i]
            while i < len(self.batch) and not s.kv_cache.resize(
                len(s.history) + len(s.input), None
            ):
                last = self.batch.pop()
                DBG_PRINT and print(f"suspending {last.kv_cache.id}")  # type: ignore[arg-type]
                last.kv_cache.suspend()
                self.suspended.append(last)

        # try to add new requests
        suspended = self.suspended
        while suspended or self.num_created < self.num_requests:
            if not suspended:
                assert self.num_created < self.num_requests
                suspended.append(self.gen_request())
            if suspended:
                step = suspended[-1]
                kv_cache = step.kv_cache
                ok = kv_cache.resume(stream)
                if ok and not self.enable_reuse and _introspection.is_commit_allowed(kv_cache):
                    kv_cache.stop_committing()
                ok = ok and kv_cache.resize(len(step.history) + len(step.input), None)
                if ok:
                    DBG_PRINT and print(f"activating {step.kv_cache.id}")  # type: ignore[arg-type]
                    self.batch.append(suspended.pop())
                else:
                    if kv_cache.status == _KVCache.Status.ACTIVE:
                        kv_cache.suspend()
                    break

        DBG_PRINT and print(  # type: ignore[arg-type]
            f"update_batch: found {len(removed)} finished requests, now with {len(self.batch)} requests"
        )

    @parameterized.expand(
        [
            (1000, 1000, 1024, True, 32, 32),
            (1000, 1000, 1024, True, 1, 32),
            (10000, 1000, 1024, True, 32, 32),
            (100, 100, 128, False, 1, 128),
            (100, 100, 128, False, 4, 64),
        ]
    )
    # @assert_no_ref_cycle
    def test_inflight_batching(
        self,
        num_requests: int,
        avg_length: int,
        gpu_quota_mb: int,
        skip_execution: bool,
        interval: int,
        tokens_per_block: int,
    ):
        self.prepare(
            gpu_quota_mb << 20, 4 << 30, 0 << 30, 36, 128, 0, tokens_per_block=tokens_per_block
        )
        self.num_requests = num_requests
        self.avg_length = avg_length
        self.interval = interval
        profile = False
        profiler = None
        if profile:
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
        tic = time.perf_counter()
        with TemporaryCudaStream([]) as s, enable_kernel_delay():
            stream = cast(CudaStream, s.handle)
            i = itertools.count()
            self.update_batch(stream)
            while self.num_finished < self.num_requests:
                DBG_PRINT and print(  # type: ignore[arg-type]
                    f"Executing batch {next(i)} with size {len(self.batch)}"
                )
                assert self.batch
                if not skip_execution:
                    self.engine.execute(self.batch, stream)
                self.update_batch(stream)
        toc = time.perf_counter()
        if profiler is not None:
            profiler.disable()
            profiler.print_stats(sort="cumtime")
            profiler.dump_stats("profiler.prof")
        if DBG_PRINT or PRINT_TIME:
            print(
                f"Time taken: {toc - tic} seconds (num_prompt_tokens: {self.acc_num_prompt_tokens}, "
                f"num_decode_tokens: {self.acc_num_decode_tokens})"
            )
        s.take_finish_event().synchronize()


class TestDisagg(TestKVCacheManagerV2):
    @parameterized.expand([512])
    # @assert_no_ref_cycle
    def test_disagg(self, prompt_len: int) -> None:
        self.prepare(128 << 20, 128 << 20, 1 << 30, 36, 128, 0)
        lora_task_id = None
        prompt = [self.next_token() for _ in range(prompt_len)]
        reuse_scope = ReuseScope(lora_id=lora_task_id)
        kv_cache = self.manager.create_kv_cache(reuse_scope, prompt)
        assert kv_cache.num_committed_tokens == 0
        with TemporaryCudaStream([]) as stream:
            success = kv_cache.resume(cast(CudaStream, stream.handle))
            assert success
            success = kv_cache.resize(prompt_len, prompt_len)
            assert success

            def transfer() -> None:
                return None

            transfer()
            kv_cache.commit(prompt)
        kv_cache.close()
        stream.take_finish_event().synchronize()


class TestDisaggregatedServing(unittest.TestCase):
    @dataclass(slots=True)
    class NodeGroup:
        class Node(NamedTuple):
            manager: KVCacheManager
            stream: CachedCudaStream
            engine: FakeEngine
            kv_cache: _KVCache

        _nodes: list[Node]
        tp_size: int

        @property
        def pp_size(self) -> int:
            return exact_div(len(self._nodes), self.tp_size)

        def __getitem__(self, key: tuple[int, int]) -> Node:
            pp_rank, tp_rank = key
            assert 0 <= pp_rank < self.pp_size and 0 <= tp_rank < self.tp_size
            return self._nodes[pp_rank * self.tp_size + tp_rank]

        def __iter__(self) -> Iterator[Node]:
            return iter(self._nodes)

        def shutdown(self) -> None:
            for node in reversed(self._nodes):
                node.kv_cache.close()
                node.manager.shutdown()

        def __init__(
            self, full_config: KVCacheManagerConfig, num_heads: int, tp_size: int, pp_size: int
        ):
            self.tp_size = tp_size
            full_layers = full_config.layers
            assert len(full_layers) % pp_size == 0
            np = tp_size * pp_size
            cache_tiers = deepcopy(full_config.cache_tiers)
            for tier in cache_tiers:
                tier.quota = tier.quota // np
            num_local_layers = len(full_layers) // pp_size
            self._nodes = []
            for pp_rank in range(pp_size):
                layer_start = num_local_layers * pp_rank
                layers = deepcopy(full_layers[layer_start : layer_start + num_local_layers])
                for layer in layers:
                    for b in layer.buffers:
                        b.size = exact_div(b.size, tp_size)
                for tp_rank in range(tp_size):
                    config = deepcopy(full_config)
                    config.cache_tiers = cache_tiers
                    config.layers = layers
                    manager = KVCacheManager(config)
                    kv_cache = manager.create_kv_cache()
                    stream = CachedCudaStream()
                    kv_cache.resume(CudaStream(stream.handle))
                    kv_cache.stop_committing()
                    engine = FakeEngine(config, exact_div(num_heads, tp_size))
                    self._nodes.append(self.Node(manager, stream, engine, kv_cache))

    _token_id_gen: Iterator[int]
    full_config: KVCacheManagerConfig
    prefill: NodeGroup
    decode: NodeGroup

    def setUp(self) -> None:
        init_cuda_once()
        self._token_id_gen = itertools.count()
        gc.collect()
        gc.disable()

    def tearDown(self) -> None:
        gc.enable()
        if hasattr(self, "decode"):
            self.decode.shutdown()
            del self.decode
        if hasattr(self, "prefill"):
            self.prefill.shutdown()
            del self.prefill

    def next_token(self) -> TokenIdExt:
        token_id = next(self._token_id_gen)
        if token_id % 100 == 99:
            return randbytes(32)
        else:
            return TokenId(token_id)

    def prepare(
        self,
        prefill_pp_size: int = 1,
        prefill_tp_size: int = 1,
        decode_pp_size: int = 1,
        decode_tp_size: int = 1,
        gpu_quota: int = 128 << 20,
        host_quota: int = 128 << 20,
        disk_quota: int = 0,
        num_layers: int = 4,
        window_size: SlidingWindowSize = 128,
        sink_tokens: int = 0,
        tokens_per_block: int = 32,
        kv_buf_size: int = 8192,
        block_quant_buf_size: int | None = None,
    ) -> None:
        assert max(prefill_tp_size, decode_tp_size) % min(prefill_tp_size, decode_tp_size) == 0
        assert max(decode_pp_size, prefill_pp_size) % min(decode_pp_size, prefill_pp_size) == 0
        self.full_config = create_config(
            tokens_per_block,
            gpu_quota,
            host_quota,
            disk_quota,
            num_layers,
            window_size,
            sink_tokens,
            kv_buf_size,
            block_quant_buf_size,
        )
        num_heads = max(prefill_tp_size, decode_tp_size)
        self.prefill = self.NodeGroup(self.full_config, num_heads, prefill_tp_size, prefill_pp_size)
        self.decode = self.NodeGroup(self.full_config, num_heads, decode_tp_size, decode_pp_size)

    def transfer(self, stream: CudaStream) -> None:
        prefill = self.prefill
        decode = self.decode
        max_pp = max(prefill.pp_size, decode.pp_size)
        max_tp = max(prefill.tp_size, decode.tp_size)

        class Slice(NamedTuple):
            num_slices: int
            slice_rank: int

        def get_rank_and_slice(max_par_size: int, par_size: int, idx: int) -> tuple[int, Slice]:
            num_slices = exact_div(max_par_size, par_size)
            rank = idx // num_slices
            slice = Slice(num_slices, idx % num_slices)
            return rank, slice

        for pp_idx in range(max_pp):
            src_pp_rank, _ = get_rank_and_slice(max_pp, prefill.pp_size, pp_idx)
            dst_pp_rank, _ = get_rank_and_slice(max_pp, decode.pp_size, pp_idx)
            layers_per_slice = exact_div(len(self.full_config.layers), max_pp)
            layers = self.full_config.layers[
                layers_per_slice * pp_idx : layers_per_slice * (pp_idx + 1)
            ]
            buffers = sum(
                ([BufferId(layer.layer_id, b.role) for b in layer.buffers] for layer in layers), []
            )
            for tp_idx in range(max_tp):
                src_tp_rank, src_tp_slice = get_rank_and_slice(max_tp, prefill.tp_size, tp_idx)
                dst_tp_rank, dst_tp_slice = get_rank_and_slice(max_tp, decode.tp_size, tp_idx)
                src = prefill[src_pp_rank, src_tp_rank]
                dst = decode[dst_pp_rank, dst_tp_rank]
                src_pages = src.manager.get_aggregated_pages(buffers)
                dst_pages = dst.manager.get_aggregated_pages(buffers)
                for src_page, dst_page in zip(src_pages, dst_pages, strict=True):
                    assert src_page.buffers == dst_page.buffers
                    assert src_page.size * prefill.tp_size == dst_page.size * decode.tp_size
                    assert (
                        dst_page.size / dst_tp_slice.num_slices
                        == src_page.size / src_tp_slice.num_slices
                    )
                    dst_indices = dst.kv_cache.get_aggregated_page_indices(
                        dst_page.layer_group_id, valid_only=True
                    )
                    src_indices = src.kv_cache.get_aggregated_page_indices(
                        src_page.layer_group_id, valid_only=True
                    )
                    need_slicing = prefill.tp_size != decode.tp_size
                    tasks = []
                    num_bytes: int
                    if not need_slicing:
                        assert src_tp_slice.num_slices == 1 and dst_tp_slice.num_slices == 1
                        num_bytes = exact_div(src_page.size, src_tp_slice.num_slices)
                        for i, j in zip(dst_indices, src_indices, strict=True):
                            task = MemToMemTask(
                                MemAddress(dst_page.base + dst_page.stride * i),
                                MemAddress(src_page.base + src_page.stride * j),
                            )
                            tasks.append(task)
                    else:
                        num_buffers = len(dst_page.buffers)
                        dst_buf_size = exact_div(dst_page.size, num_buffers)
                        src_buf_size = exact_div(src_page.size, num_buffers)
                        num_bytes = exact_div(dst_buf_size, dst_tp_slice.num_slices)
                        assert num_bytes == exact_div(src_buf_size, src_tp_slice.num_slices)
                        for i, j in zip(dst_indices, src_indices, strict=True):
                            dst_base = (
                                dst_page.base
                                + dst_page.stride * i
                                + num_bytes * dst_tp_slice.slice_rank
                            )
                            src_base = (
                                src_page.base
                                + src_page.stride * j
                                + num_bytes * src_tp_slice.slice_rank
                            )
                            for b in range(num_buffers):
                                task = MemToMemTask(
                                    MemAddress(dst_base + dst_buf_size * b),
                                    MemAddress(src_base + src_buf_size * b),
                                )
                                tasks.append(task)
                    copy_device_to_device(tasks, num_bytes, stream)

    @parameterized.expand([(1, 1, 1, 1), (1, 2, 1, 1), (1, 1, 1, 2), (2, 1, 1, 1), (1, 1, 2, 1)])
    def test_disaggregated_serving(
        self,
        prefill_pp_size: int,
        prefill_tp_size: int,
        decode_pp_size: int,
        decode_tp_size: int,
    ) -> None:
        self.prepare(prefill_pp_size, prefill_tp_size, decode_pp_size, decode_tp_size)

        prompt_len = 185
        prompt = [self.next_token() for _ in range(prompt_len)]
        for node in self.prefill:
            node.kv_cache.capacity = prompt_len
            node.engine.execute(
                [Step(node.kv_cache, prompt, [])], cast(CudaStream, node.stream.handle)
            )
            node.kv_cache.history_length = prompt_len
        for node in self.decode:
            node.kv_cache.resize(prompt_len, prompt_len)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            # make both prefill and decode pages available in the steam used for data copy
            for src in self.prefill:
                src.kv_cache.cuda_stream = stream
            for dst in self.decode:
                dst.kv_cache.cuda_stream = stream
            # Do that data transfer
            self.transfer(stream)
        # OK to close the prefill requests now.
        for node in self.prefill:
            node.kv_cache.close()
        _ = s.take_finish_event()  # no need to synchronize.
        # ref-check from decode nodes.
        for node in self.decode:
            stream = cast(CudaStream, node.stream.handle)
            node.kv_cache.cuda_stream = stream
            node.engine.execute([Step(node.kv_cache, [], prompt)], stream)
        for node in self.decode:
            node.stream.synchronize()
            node.kv_cache.close()


class TestComplexModels(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()

    def tearDown(self) -> None:
        gc.enable()

    def test_complex_model_0(self) -> None:
        role = DataRole("buf0")
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=role, size=131072)],
                sliding_window_size=128,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(role=role, size=131072)],
                sliding_window_size=128,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(2),
                buffers=[BufferConfig(role=role, size=98304)],
                sliding_window_size=None,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(3),
                buffers=[BufferConfig(role=role, size=163840)],
                sliding_window_size=64,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(4),
                buffers=[BufferConfig(role=role, size=163840)],
                sliding_window_size=64,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(5),
                buffers=[BufferConfig(role=role, size=65536)],
                sliding_window_size=None,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(6),
                buffers=[BufferConfig(role=role, size=131072)],
                sliding_window_size=64,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(7),
                buffers=[BufferConfig(role=role, size=131072)],
                sliding_window_size=64,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(8),
                buffers=[BufferConfig(role=role, size=131072)],
                sliding_window_size=128,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(9),
                buffers=[BufferConfig(role=role, size=32768)],
                sliding_window_size=None,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(10),
                buffers=[BufferConfig(role=role, size=262144)],
                sliding_window_size=128,
                num_sink_tokens=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(11),
                buffers=[BufferConfig(role=role, size=262144)],
                sliding_window_size=128,
                num_sink_tokens=None,
            ),
        ]

        config = KVCacheManagerConfig(
            tokens_per_block=128,
            cache_tiers=[
                GpuCacheTierConfig(quota=1024 * 1024 * 1024),
                HostCacheTierConfig(quota=8000 << 20),
            ],
            max_util_for_resume=0.95,
            layers=layers,
        )
        manager = KVCacheManager(config)
        del manager

    def test_complex_model_1(self) -> None:
        """Regression: large slot_size PGs with low slot_cnt caused deadloop."""
        role = DataRole("key")
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=role, size=65536)],
                sliding_window_size=128,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(role=role, size=65536)],
                sliding_window_size=128,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(2),
                buffers=[BufferConfig(role=role, size=16384)],
                sliding_window_size=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(3),
                buffers=[BufferConfig(role=role, size=524288)],
                sliding_window_size=8,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(4),
                buffers=[BufferConfig(role=role, size=524288)],
                sliding_window_size=8,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(5),
                buffers=[BufferConfig(role=role, size=4224)],
                sliding_window_size=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(6),
                buffers=[BufferConfig(role=role, size=131072)],
                sliding_window_size=8,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(7),
                buffers=[BufferConfig(role=role, size=131072)],
                sliding_window_size=8,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(8),
                buffers=[BufferConfig(role=role, size=65536)],
                sliding_window_size=128,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(9),
                buffers=[BufferConfig(role=role, size=512)],
                sliding_window_size=None,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(10),
                buffers=[BufferConfig(role=role, size=262144)],
                sliding_window_size=128,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(11),
                buffers=[BufferConfig(role=role, size=262144)],
                sliding_window_size=128,
            ),
        ]

        typical_step = BatchDesc(
            kv_caches=[KVCacheDesc(capacity=4197, history_length=4196)] * 3,
        )
        constraints = [
            BatchDesc([KVCacheDesc(capacity=4197, history_length=0)]),
            BatchDesc([KVCacheDesc(capacity=7168, history_length=0)]),
        ]

        config = KVCacheManagerConfig(
            tokens_per_block=128,
            cache_tiers=[GpuCacheTierConfig(quota=212549334)],
            layers=layers,
            typical_step=typical_step,
            constraints=constraints,
        )
        manager = KVCacheManager(config)
        del manager


class TestResizeQuota(TestKVCacheManagerV2):
    def test_resize_quota(self) -> None:
        self.prepare(64 << 20, 128 << 20, 128 << 20, 36, 128, 1, kv_buf_size=32768)
        # if we have n blocks, we need 8192*2*18*(1+5+n) bytes of memory. For the (1+5+n), 1 is for sink
        # blocks, 5 is for SWA (window=128), n is for full attention.
        max_seq_len = 32 * 22  # 23 blocks will require more than 32MB memory
        seq_len = max_seq_len
        tokens_per_block = self.cfg.tokens_per_block
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        # First commit some blocks to fill all levels of cache. This helps test the case where shrinking
        # the quota will drop some pages from the last-level cache.
        for _ in range(11):
            kv_cache = self.manager.create_kv_cache()
            kv_cache.resume(stream)
            for i in range(exact_div(seq_len, tokens_per_block)):
                kv_cache.capacity = tokens_per_block * (i + 1)
                input = [self.next_token() for _ in range(tokens_per_block)]
                kv_cache.commit(input)
            kv_cache.close()

        # Now create two requests.
        kv_cache_lst = [self.manager.create_kv_cache() for _ in range(2)]
        for kv_cache in kv_cache_lst:
            success = kv_cache.resume(stream)
            assert success
            kv_cache.stop_committing()
            success = kv_cache.resize(seq_len, seq_len)
            assert success
        # Without reversed, we will hit a corner case where all cache levels are
        # full, but the kv cache we want to resume is in the last level, while
        # the gpu cache level is occupied by the request we don't resume first.
        # Then we have a dead lock.
        # To fix this, we need to have a fallback non-batched iterative page
        # migration strategy instead of batched_lock_to_gpu. But this happens
        # only in very rare case, where the last-level cache can't hold all
        # suspended requests, and resume happens in FIFO order.
        for kv_cache in reversed(kv_cache_lst):
            kv_cache.suspend()
        GPU_LEVEL = CacheLevel(0)
        HOST_LEVEL = CacheLevel(1)
        DISK_LEVEL = CacheLevel(2)
        # Shrink the gpu quota
        success = self.manager.resize(GPU_LEVEL, 32 << 20)
        assert success and self.manager.get_quota(GPU_LEVEL) <= 32 << 20
        # also shrink the host quota, this would evict some pages to disk.
        # 16MB is the smallest shrink that still satisfies the SWA worst-case
        # min_slots floor (sink + window blocks across all pool groups).
        success = self.manager.resize(HOST_LEVEL, 16 << 20)
        assert success and self.manager.get_quota(HOST_LEVEL) <= 16 << 20
        # also shrink the disk quota, this would drop some old pages
        success = self.manager.resize(DISK_LEVEL, 32 << 20)
        assert success and self.manager.get_quota(DISK_LEVEL) <= 32 << 20
        success = kv_cache_lst[0].resume(stream)
        assert success
        # After shrinking, GPU memory can hold only one request, so expect failure
        # for resuming of the second request.
        success = kv_cache_lst[1].resume(stream)
        assert not success

        kv_cache_lst[0].suspend()
        # Expand it back to the original size
        success = self.manager.resize(GPU_LEVEL, 64 << 20)
        assert success
        success = self.manager.resize(HOST_LEVEL, 128 << 20)
        assert success
        prefetch_target = kv_cache_lst[1]
        # _introspection.active_page_stats returns (active counts, unscheduled evictable counts) by cache level.
        prefetch_counts_before, _ = _introspection.active_page_stats(prefetch_target)
        self.assertGreater(prefetch_counts_before[DISK_LEVEL], 0)
        success = prefetch_target.prefetch(HOST_LEVEL)
        self.assertEqual(success, True)
        prefetch_counts_after, unscheduled_evictable_after = _introspection.active_page_stats(
            prefetch_target
        )
        self.assertEqual(prefetch_counts_after[GPU_LEVEL], prefetch_counts_before[GPU_LEVEL])
        self.assertEqual(prefetch_counts_after[DISK_LEVEL], 0)
        self.assertEqual(
            prefetch_counts_after[HOST_LEVEL],
            prefetch_counts_before[HOST_LEVEL] + prefetch_counts_before[DISK_LEVEL],
        )
        self.assertEqual(unscheduled_evictable_after[HOST_LEVEL], 0)
        # The disk-prefetch counter measures what this call moved: every page that was on disk
        # beforehand, counted the way iterOnboardBlocks/iterOffloadBlocks count.
        moved_blocks = self.manager.get_and_reset_iteration_disk_prefetch_blocks()
        self.assertEqual(moved_blocks, prefetch_counts_before[DISK_LEVEL])
        # Nothing is left on disk now, so a second prefetch moves nothing and counts nothing.
        self.assertTrue(prefetch_target.prefetch(HOST_LEVEL))
        self.assertEqual(self.manager.get_and_reset_iteration_disk_prefetch_blocks(), 0)
        # Now both requests can resume
        for kv_cache in kv_cache_lst:
            success = kv_cache.resume(stream)
            assert success

        for kv_cache in kv_cache_lst:
            kv_cache.close()
        self.manager.shutdown()


class TestHeteroTokensPerBlock(TestKVCacheManagerV2):
    def test_hetero_tokens_per_block(self) -> None:
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[
                    BufferConfig(role=Role.KEY, size=131072),
                    BufferConfig(role=Role.VALUE, size=131072),
                ],
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[
                    BufferConfig(role=Role.KEY, size=131072, tokens_per_block_override=64),
                    BufferConfig(role=Role.VALUE, size=131072, tokens_per_block_override=64),
                ],
            ),
        ]
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=128,
            cache_tiers=[
                GpuCacheTierConfig(quota=256 << 20),
                HostCacheTierConfig(quota=1 << 30),
            ],
            layers=layers,
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)
        kv_cache = self.manager.create_kv_cache()
        prompt_len = 163
        prompt = [self.next_token() for _ in range(prompt_len)]
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache.resume(stream)
        kv_cache.capacity = prompt_len
        history = []
        input = prompt
        self.engine.execute([Step(kv_cache, input, history)], stream)
        kv_cache.commit(input)
        history.extend(input)
        decode_len = 97
        for _ in range(decode_len):
            kv_cache.capacity = len(history) + 1
            input = [self.next_token()]
            self.engine.execute([Step(kv_cache, input, history)], stream)
            kv_cache.commit(input)
            history.extend(input)
        kv_cache.close()

        # test reuse.
        second_prompt_len = 79
        prompt = history + [self.next_token() for _ in range(second_prompt_len)]
        kv_cache = self.manager.create_kv_cache(None, prompt)
        kv_cache.resume(stream)
        assert kv_cache.num_committed_tokens == len(history)
        # empty input just for ref-check.
        input = []
        self.engine.execute([Step(kv_cache, input, history)], stream)
        kv_cache.close()


class TestKVCacheReusePerformance(TestKVCacheManagerV2):
    """Test class for measuring KV cache reuse performance."""

    def test_cache_reuse_performance(self, profile: bool = False) -> None:
        """Performance test for KV cache reuse (prefill only).

        - First pass: 20 requests with 1000 tokens per prompt (cold cache).
        - Second pass: Re-run the same 20 requests to achieve 100% cache hit rate.
        """
        self.prepare(
            gpu_quota=512 << 20,
            host_quota=512 << 20,
            disk_quota=1 << 30,
            num_layers=36,
            window_size=None,
            sink_tokens=0,
            tokens_per_block=32,
            kv_buf_size=8192,
        )

        num_requests = 20
        prompt_len = 1000

        prompts = []
        for _ in range(num_requests):
            prompt = [self.next_token() for _ in range(prompt_len)]
            prompts.append(prompt)

        def run_requests(prompts: list[list[TokenIdExt]]) -> dict:
            """Run all requests (prefill only) and return performance metrics."""
            results = {
                "total_time": 0.0,
                "num_reused_tokens": 0,
                "num_computed_tokens": 0,
            }

            tic_total = time.perf_counter()

            with TemporaryCudaStream([]) as s:
                stream = cast(CudaStream, s.handle)

                requests = []

                for req_id, prompt in enumerate(prompts):
                    kv_cache = self.manager.create_kv_cache(None, prompt)
                    num_reused = kv_cache.num_committed_tokens

                    success = kv_cache.resume(stream)
                    assert success, f"Failed to resume cache for request {req_id}"

                    results["num_reused_tokens"] += num_reused
                    results["num_computed_tokens"] += prompt_len - num_reused

                    if not kv_cache.resize(prompt_len + 1):
                        raise OutOfPagesError(f"Not enough pages for request {req_id}")

                    input_tokens = prompt[num_reused:]

                    requests.append(Step(kv_cache, input_tokens, prompt[:num_reused]))

                for r in requests:
                    r.kv_cache.commit(r.input)
                    r.kv_cache.close()

            s.take_finish_event().synchronize()

            toc_total = time.perf_counter()
            results["total_time"] = toc_total - tic_total

            return results

        profiler1 = None
        profiler2 = None
        if profile:
            import cProfile

            profiler1 = cProfile.Profile()
            profiler2 = cProfile.Profile()

        # First pass: No cache reuse expected
        if profiler1 is not None:
            profiler1.enable()
        run_requests(prompts)
        if profiler1 is not None:
            profiler1.disable()

        # Second pass: 100% cache reuse expected
        if profiler2 is not None:
            profiler2.enable()
        results_pass2 = run_requests(prompts)
        if profiler2 is not None:
            profiler2.disable()

        if PRINT_TIME:
            print(f"total_time = {results_pass2['total_time']}")

        # Verify 100% hit rate on second pass
        total_tokens_pass2 = (
            results_pass2["num_reused_tokens"] + results_pass2["num_computed_tokens"]
        )
        actual_hit_rate = (
            (results_pass2["num_reused_tokens"] / total_tokens_pass2 * 100)
            if total_tokens_pass2 > 0
            else 0
        )
        assert abs(actual_hit_rate - 100.0) < 0.01, (
            f"Expected 100% hit rate on second pass, got {actual_hit_rate:.2f}%"
        )

        if profile:
            profiler1.print_stats(sort="cumtime")
            profiler2.print_stats(sort="cumtime")
            profiler1.dump_stats("kv_cache_reuse_pass1.prof")
            profiler2.dump_stats("kv_cache_reuse_pass2.prof")


class TestSSMSupport(unittest.TestCase):
    """Tests for basic SSM (State Space Model / Mamba) support in KVCacheManager v2."""

    _token_id_gen: Iterator[int]

    def setUp(self) -> None:
        init_cuda_once()
        self._token_id_gen = itertools.count()
        gc.collect()
        gc.disable()

    def tearDown(self) -> None:
        gc.enable()
        if hasattr(self, "manager"):
            self.manager.shutdown()
            del self.manager

    def next_token(self) -> TokenIdExt:
        return TokenId(next(self._token_id_gen))

    def _make_ssm_config(
        self,
        tokens_per_block: int = 32,
        gpu_quota: int = 32 << 20,
        num_attn_layers: int = 2,
        num_ssm_layers: int = 2,
        window_size: SlidingWindowSize = None,
        commit_min_snapshot: bool = True,
        enable_partial_reuse: bool = False,
    ) -> KVCacheManagerConfig:
        layers = []
        lid = 0
        for _ in range(num_attn_layers):
            layers.append(
                AttentionLayerConfig(
                    layer_id=LayerId(lid),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=8192),
                        BufferConfig(role=DataRole("value"), size=8192),
                    ],
                    sliding_window_size=window_size,
                )
            )
            lid += 1
        for _ in range(num_ssm_layers):
            layers.append(
                SsmLayerConfig(
                    layer_id=LayerId(lid),
                    buffers=[
                        BufferConfig(role=DataRole("ssm_state"), size=8192),
                    ],
                )
            )
            lid += 1
        return KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=layers,
            enable_partial_reuse=enable_partial_reuse,
            commit_min_snapshot=commit_min_snapshot,
        )

    def test_suspend_and_resume_with_ssm(self) -> None:
        """Suspend and resume work correctly (SSM page locks/unlocks)."""
        cfg = self._make_ssm_config()
        self.manager = KVCacheManager(cfg)
        kv_cache = self.manager.create_kv_cache()
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache.resume(stream)
        # Find the SSM layer group ID from the config.
        ssm_lg = None
        for layer in cfg.layers:
            if isinstance(layer, SsmLayerConfig):
                ssm_lg = self.manager.get_layer_group_id(layer.layer_id)
                break
        assert ssm_lg is not None
        # Grow some capacity
        kv_cache.capacity = 100
        initial_slot = kv_cache.get_ssm_block_base_index(ssm_lg)
        self.assertNotEqual(initial_slot, BAD_PAGE_INDEX)
        # Suspend
        kv_cache.stop_committing()
        kv_cache.suspend()
        self.assertEqual(kv_cache.status, _KVCache.Status.SUSPENDED)
        # Resume
        success = kv_cache.resume(stream)
        self.assertTrue(success)
        self.assertEqual(kv_cache.status, _KVCache.Status.ACTIVE)
        # SSM slot should be the same
        resumed_slot = kv_cache.get_ssm_block_base_index(ssm_lg)
        self.assertEqual(initial_slot, resumed_slot, "SSM slot unchanged after suspend/resume")
        kv_cache.close()

    def test_no_reuse_with_ssm(self) -> None:
        """input_tokens are accepted but no prefix reuse happens without a prior snapshot."""
        cfg = self._make_ssm_config(tokens_per_block=32)
        self.manager = KVCacheManager(cfg)
        # No request has committed these tokens yet, so there is no SSM snapshot to reuse.
        tokens = [self.next_token() for _ in range(64)]
        kv_cache = self.manager.create_kv_cache(input_tokens=tokens)
        self.assertEqual(kv_cache.num_committed_tokens, 0, "No reuse before first snapshot")
        # Resume before close so cuda_stream is set
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache.resume(stream)
        kv_cache.close()

    @parameterized.expand(
        [
            ("miss", None, 48, False, (1, 0, 1, 0, 48, 0, 0)),
            ("aligned_hit", 32, 48, False, (1, 1, 0, 32, 16, 1, 0)),
            ("unaligned_hit", 48, 64, True, (1, 1, 0, 48, 16, 0, 1)),
        ]
    )
    def test_ssm_snapshot_iteration_stats(
        self,
        _name: str,
        snapshot_length: int | None,
        lookup_length: int,
        enable_partial_reuse: bool,
        expected: tuple[int, int, int, int, int, int, int],
    ) -> None:
        tokens_per_block = 32
        cfg = self._make_ssm_config(
            tokens_per_block=tokens_per_block,
            enable_partial_reuse=enable_partial_reuse,
        )
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(lookup_length)]

        if snapshot_length is not None:
            seed = self.manager.create_kv_cache()
            seed.resume(stream)
            seed.capacity = snapshot_length
            seed.history_length = snapshot_length
            seed.commit(prompt[:snapshot_length], is_end=True)
            seed.close()

        reused = self.manager.create_kv_cache(
            input_tokens=prompt,
            id=101,
            # This is only a sizing hint; lookup telemetry must use the
            # actual input_tokens length.
            expected_prompt_length=lookup_length + 17,
        )
        self.assertEqual(reused.num_committed_tokens, expected[3])
        self.assertEqual(self.manager.get_dirty_stats_kv_cache_ids(), {101})
        reused.commit_pending_stats()
        self.assertEqual(self.manager.get_dirty_stats_kv_cache_ids(), set())

        ssm_life_cycle_id = _introspection.ssm_life_cycle_id(self.manager)
        assert ssm_life_cycle_id is not None
        snapshot_stats = self.manager.get_and_reset_ssm_snapshot_iteration_stats()
        self.assertEqual(set(snapshot_stats), {ssm_life_cycle_id})
        stats = snapshot_stats[ssm_life_cycle_id]
        self.assertEqual(
            (
                stats.iter_snapshot_lookups,
                stats.iter_snapshot_hits,
                stats.iter_snapshot_misses,
                stats.iter_reused_tokens,
                stats.iter_unreused_tokens,
                stats.iter_aligned_snapshot_hits,
                stats.iter_unaligned_snapshot_hits,
            ),
            expected,
        )
        self.assertEqual(stats.iter_snapshot_hit_rate, expected[1] / expected[0])
        self.assertEqual(self.manager.get_and_reset_ssm_snapshot_iteration_stats(), {})

        reused.resume(stream)
        reused.close()

    def test_discard_ssm_snapshot_stats_clears_dirty_state(self) -> None:
        cfg = self._make_ssm_config()
        self.manager = KVCacheManager(cfg)
        tokens = [self.next_token() for _ in range(16)]

        kv_cache = self.manager.create_kv_cache(input_tokens=tokens, id=101)
        self.assertEqual(self.manager.get_dirty_stats_kv_cache_ids(), {101})
        kv_cache.discard_pending_stats()

        self.assertEqual(self.manager.get_dirty_stats_kv_cache_ids(), set())
        self.assertEqual(self.manager.get_and_reset_ssm_snapshot_iteration_stats(), {})
        stream_holder = CachedCudaStream()
        kv_cache.resume(cast(CudaStream, stream_holder.handle))
        kv_cache.close()

    def test_ssm_resume_records_intra_device_copy(self) -> None:
        """The SSM deferred copy on resume is counted in iteration stats.

        First resume of a cache reusing an SSM snapshot copies the snapshot
        into a private slot; the copy must appear in the SSM life cycle's
        iteration stats (TRTLLM-15217). Offload, onboard and host-drop for the
        same life cycle are covered by KvCacheManagerV2StatsTest.
        """
        tokens_per_block = 32
        cfg = self._make_ssm_config(tokens_per_block=tokens_per_block)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(48)]

        seed = self.manager.create_kv_cache()
        seed.resume(stream)
        seed.capacity = tokens_per_block
        seed.history_length = tokens_per_block
        seed.commit(prompt[:tokens_per_block], is_end=True)
        seed.close()

        reused = self.manager.create_kv_cache(input_tokens=prompt, id=101)
        self.assertEqual(reused.num_committed_tokens, tokens_per_block)
        reused.commit_pending_stats()
        # Drop everything recorded so far; only the resume below should count.
        self.manager.get_and_reset_ssm_snapshot_iteration_stats()
        self.manager.get_and_reset_iteration_stats()

        self.assertTrue(reused.resume(stream))
        ssm_life_cycle_id = _introspection.ssm_life_cycle_id(self.manager)
        assert ssm_life_cycle_id is not None
        stats = self.manager.get_and_reset_iteration_stats()
        self.assertIn(ssm_life_cycle_id, stats)
        self.assertEqual(stats[ssm_life_cycle_id].iter_intra_device_copy_blocks, 1)
        self.assertGreater(stats[ssm_life_cycle_id].iter_intra_device_copy_bytes, 0)
        reused.close()

    def test_ssm(self) -> None:
        """Inference with SSM layer: prefill 63 tokens, decode 52 tokens."""
        cfg = self._make_ssm_config()
        self.manager = KVCacheManager(cfg)
        engine = FakeEngine(cfg)
        kv_cache = self.manager.create_kv_cache()
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache.resume(stream)
        kv_cache.stop_committing()
        # prefill
        prompt = [self.next_token() for _ in range(63)]
        kv_cache.capacity = len(prompt)
        kv_cache.history_length = len(prompt)
        engine.execute([Step(kv_cache, prompt, [])], stream)
        history = list(prompt)
        # decode
        for _ in range(52):
            kv_cache.capacity = len(history) + 1
            token = self.next_token()
            engine.execute([Step(kv_cache, [token], history)], stream)
            history.append(token)
            kv_cache.history_length = len(history)
        # final check
        engine.execute([Step(kv_cache, [], history)], stream)
        kv_cache.close()

    def _make_ssm_reuse_config(
        self,
        tokens_per_block: int = 32,
        gpu_quota: int = 32 << 20,
        num_attn_layers: int = 2,
        num_ssm_layers: int = 2,
    ) -> KVCacheManagerConfig:
        return self._make_ssm_config(
            tokens_per_block=tokens_per_block,
            gpu_quota=gpu_quota,
            num_attn_layers=num_attn_layers,
            num_ssm_layers=num_ssm_layers,
        )

    def test_ssm_reuse_snapshots_each_commit(self) -> None:
        """SSM keeps reusable snapshots at committed prefix lengths."""
        cfg = self._make_ssm_reuse_config(tokens_per_block=32)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        prompt = [self.next_token() for _ in range(128)]
        early_prompt = prompt[:96]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = len(early_prompt)
        kv1.history_length = len(early_prompt)
        kv1.commit(early_prompt)
        kv1.stop_committing()
        kv1.close()

        kv2 = self.manager.create_kv_cache(input_tokens=early_prompt)
        self.assertEqual(kv2.num_committed_tokens, len(early_prompt))
        kv2.resume(stream)
        kv2.close()

        kv3 = self.manager.create_kv_cache()
        kv3.resume(stream)
        kv3.capacity = len(prompt)
        kv3.history_length = len(prompt)
        kv3.commit(prompt)
        kv3.stop_committing()
        kv3.close()

        kv4 = self.manager.create_kv_cache(input_tokens=prompt)
        self.assertEqual(kv4.num_committed_tokens, len(prompt))
        kv4.resume(stream)
        kv4.close()

        kv5 = self.manager.create_kv_cache(input_tokens=early_prompt)
        self.assertEqual(kv5.num_committed_tokens, len(early_prompt))
        kv5.resume(stream)
        kv5.close()

    def test_ssm_reuse_data_integrity(self) -> None:
        """After reuse, SSM data matches the snapshot (verified by FakeEngine)."""
        tokens_per_block = 32
        cfg = self._make_ssm_reuse_config(tokens_per_block=tokens_per_block)
        self.manager = KVCacheManager(cfg)
        engine = FakeEngine(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        # Request 1: prefill and commit
        prompt = [self.next_token() for _ in range(128)]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = len(prompt)
        kv1.history_length = len(prompt)
        engine.execute([Step(kv1, prompt, [])], stream)
        kv1.commit(prompt)
        kv1.stop_committing()
        kv1.close()

        # Request 2: reuse and verify data integrity
        kv2 = self.manager.create_kv_cache(input_tokens=prompt)
        kv2.resume(stream)
        # Grow capacity to match prompt
        kv2.capacity = len(prompt)
        kv2.history_length = len(prompt)
        # Check that the reused data is valid (FakeEngine verifies page contents)
        engine.execute([Step(kv2, [], prompt)], stream)
        # Decode some tokens on top
        history = list(prompt)
        for _ in range(10):
            kv2.capacity = len(history) + 1
            token = self.next_token()
            engine.execute([Step(kv2, [token], history)], stream)
            history.append(token)
            kv2.history_length = len(history)
        kv2.close()

    def test_ssm_reuse_keeps_snapshots_from_multiple_commits(self) -> None:
        """Multiple commit() calls keep independently reusable SSM snapshots."""
        cfg = self._make_ssm_reuse_config(tokens_per_block=32)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        prompt = [self.next_token() for _ in range(96)]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = 32
        kv1.commit(prompt[:32])

        kv1.capacity = 64
        kv1.commit(prompt[32:64])
        kv1.close()

        kv2 = self.manager.create_kv_cache(input_tokens=prompt[:32])
        self.assertEqual(kv2.num_committed_tokens, 32)
        kv2.resume(stream)
        kv2.close()

        kv3 = self.manager.create_kv_cache(input_tokens=prompt[:48])
        self.assertEqual(kv3.num_committed_tokens, 32)
        kv3.resume(stream)
        kv3.close()

        kv4 = self.manager.create_kv_cache(input_tokens=prompt)
        self.assertEqual(kv4.num_committed_tokens, 64)
        kv4.resume(stream)
        kv4.close()

    def test_num_reusable_tokens_before_hybrid_pruning_isolates_recurrent_truncation(
        self,
    ) -> None:
        """The diagnostic separates a short attention match from recurrent pruning.

        Partial reuse is required for the two numbers to differ at all: without
        it a match is block-aligned, so the attention-only prefix and the final
        committed prefix are cut at the same block boundary and the diagnostic
        is indistinguishable from num_committed_tokens.
        """
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        prompt = [self.next_token() for _ in range(96)]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = 32
        kv1.commit(prompt[:32])
        kv1.capacity = 64
        kv1.commit(prompt[32:64])
        kv1.close()

        # Attention pages partially cover all 48 lookup tokens, but the latest
        # reusable SSM snapshot sits at 32 — so recurrent pruning, not a short
        # attention match, is what cut the reuse.
        kv = self.manager.create_kv_cache(input_tokens=prompt[:48])
        self.assertEqual(kv.num_committed_tokens, 32)
        self.assertEqual(kv._get_num_reusable_tokens_before_hybrid_pruning(), 48)
        kv.resume(stream)
        kv.close()

        # When the snapshot and the attention match agree, the diagnostic must
        # collapse onto num_committed_tokens rather than reporting the lookup.
        kv = self.manager.create_kv_cache(input_tokens=prompt[:64])
        self.assertEqual(kv.num_committed_tokens, 64)
        self.assertEqual(kv._get_num_reusable_tokens_before_hybrid_pruning(), 64)
        kv.resume(stream)
        kv.close()

    def test_first_new_block_probe_uses_snapshot_after_backoff(self) -> None:
        """Snapshot pruning determines which already-present block needs recomputation."""
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        cfg.reuse_match_backoff = 1
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(64)]
        kv = self.manager.create_kv_cache()
        self.assertTrue(kv.resume(stream))
        kv.capacity = 32
        kv.commit(prompt[:32])
        kv.capacity = 64
        kv.commit(prompt[32:])
        kv.close()

        # Backoff reduces the attention match to 63; the last eligible SSM
        # snapshot is at 32. The second full block already has a tree key but
        # needs new computation, so an absent-key-only probe is insufficient.
        self.assertEqual(self.manager.probe_reuse(None, prompt), 32)
        keys = list(sequence_to_blockchain_keys(32, ReuseScope(), prompt))
        self.assertEqual(self.manager.probe_first_new_block_key(None, prompt), keys[2][1])
        claimed = self.manager.create_kv_cache(input_tokens=prompt)
        self.assertEqual(claimed.num_committed_tokens, 32)
        self.assertTrue(claimed.resume(stream))
        claimed.close()

    def test_reuse_match_reports_content_divergence_depth(self) -> None:
        """The raw walk depth locates the fork, even where pruning shortens reuse."""
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        prompt = [self.next_token() for _ in range(96)]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = 32
        kv1.commit(prompt[:32])
        kv1.capacity = 64
        kv1.commit(prompt[32:64])
        kv1.close()

        # Duplicate lookup: the whole range matched, so there is no fork and the
        # divergence depth saturates at the lookup length.
        duplicate = self.manager.create_kv_cache(input_tokens=prompt[:64])
        self.assertEqual(duplicate._get_num_reusable_tokens_before_pruning(), 64)
        self.assertEqual(duplicate._get_num_reusable_tokens_before_hybrid_pruning(), 64)
        duplicate.resume(stream)
        duplicate.close()

        # Diverging lookup: content splits after the committed prefix, so the
        # divergence depth stays below the lookup length.
        sibling = prompt[:64] + [self.next_token() for _ in range(32)]
        forked = self.manager.create_kv_cache(input_tokens=sibling)
        self.assertEqual(forked._get_num_reusable_tokens_before_pruning(), 64)
        self.assertLess(forked._get_num_reusable_tokens_before_pruning(), len(sibling))
        forked.resume(stream)
        forked.close()

        # Snapshot pruning shortens reuse to the last usable SSM snapshot, but
        # leaves the divergence depth untouched: attention matched all 48.
        partial = self.manager.create_kv_cache(input_tokens=prompt[:48])
        self.assertEqual(partial._get_num_reusable_tokens_before_pruning(), 48)
        self.assertEqual(partial._get_num_reusable_tokens_before_hybrid_pruning(), 48)
        self.assertEqual(partial.num_committed_tokens, 32)
        partial.resume(stream)
        partial.close()

    def test_reuse_match_depths_are_monotonically_ordered(self) -> None:
        """Pruning only ever shortens: reuse <= hybrid depth <= divergence depth."""
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        prompt = [self.next_token() for _ in range(128)]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        for i in range(3):
            kv1.capacity = 32 * (i + 1)
            kv1.commit(prompt[32 * i : 32 * (i + 1)])
        kv1.close()

        for lookup_len in (16, 32, 48, 64, 80, 96, 112, 128):
            kv = self.manager.create_kv_cache(input_tokens=prompt[:lookup_len])
            divergence = kv._get_num_reusable_tokens_before_pruning()
            hybrid = kv._get_num_reusable_tokens_before_hybrid_pruning()
            self.assertLessEqual(kv.num_committed_tokens, hybrid, msg=f"len={lookup_len}")
            self.assertLessEqual(hybrid, divergence, msg=f"len={lookup_len}")
            self.assertLessEqual(divergence, lookup_len, msg=f"len={lookup_len}")
            kv.resume(stream)
            kv.close()

    def test_branch_snapshot_at_divergence_unlocks_sibling_reuse(self) -> None:
        """A snapshot at the fork lets later siblings reuse the shared prefix.

        This is the R1-R3 sequence the branch-snapshot feature targets: R1
        snapshots only at its own prompt end, which is past the fork and so
        useless to siblings. R2 snapshots at the divergence depth instead, and
        R3 then reuses the whole shared prefix.
        """
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        shared = [self.next_token() for _ in range(64)]
        r1_prompt = shared + [self.next_token() for _ in range(32)]
        r2_prompt = shared + [self.next_token() for _ in range(32)]
        r3_prompt = shared + [self.next_token() for _ in range(32)]

        # R1 commits once, so its only snapshot sits at its prompt end.
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = len(r1_prompt)
        kv1.commit(r1_prompt)
        kv1.close()

        # A sibling locates the fork at 64 but reuses nothing: the sole snapshot
        # is at 96, past the point where the content diverges.
        probe = self.manager.create_kv_cache(input_tokens=r3_prompt)
        self.assertEqual(probe._get_num_reusable_tokens_before_pruning(), 64)
        self.assertEqual(probe._get_num_reusable_tokens_before_hybrid_pruning(), 64)
        self.assertEqual(probe.num_committed_tokens, 0)
        probe.resume(stream)
        probe.close()

        # R2 diverges at the same depth and snapshots there before continuing.
        kv2 = self.manager.create_kv_cache(input_tokens=r2_prompt)
        self.assertEqual(kv2._get_num_reusable_tokens_before_pruning(), 64)
        kv2.resume(stream)
        kv2.capacity = 64
        kv2.commit(r2_prompt[:64])
        kv2.capacity = len(r2_prompt)
        kv2.commit(r2_prompt[64:])
        kv2.close()

        # R3 now reuses the full shared prefix instead of re-prefilling it.
        kv3 = self.manager.create_kv_cache(input_tokens=r3_prompt)
        self.assertEqual(kv3.num_committed_tokens, 64)
        kv3.resume(stream)
        kv3.close()

    def test_reuse_match_divergence_is_zero_without_a_shared_prefix(self) -> None:
        """A lookup that shares nothing reports a zero divergence depth."""
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        prompt = [self.next_token() for _ in range(64)]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = 32
        kv1.commit(prompt[:32])
        kv1.close()

        unrelated = [self.next_token() for _ in range(64)]
        kv2 = self.manager.create_kv_cache(input_tokens=unrelated)
        self.assertEqual(kv2._get_num_reusable_tokens_before_pruning(), 0)
        self.assertEqual(kv2.num_committed_tokens, 0)
        kv2.resume(stream)
        kv2.close()

    def test_ssm_planned_drop_targets_latest_snapshot_with_shared_plans(self) -> None:
        """Shared plans drop only their conversation endpoint snapshot."""
        cfg = self._make_ssm_config(tokens_per_block=32)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(64)]

        kv_cache = self.manager.create_kv_cache()
        kv_cache.resume(stream)
        kv_cache.capacity = 32
        kv_cache.commit(prompt[:32])
        kv_cache.capacity = 64
        kv_cache.commit(prompt[32:])
        kv_cache.stop_committing()
        first_handle = kv_cache.plan_committed_block_drop()
        second_handle = kv_cache.plan_committed_block_drop()
        self.assertIsNotNone(first_handle)
        self.assertIsNotNone(second_handle)
        kv_cache.close()

        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 64)
        assert first_handle is not None
        first_handle.drop()
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 64)
        assert second_handle is not None
        second_handle.drop()
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 32)

        empty_cache = self.manager.create_kv_cache()
        empty_cache.resume(stream)
        empty_cache.stop_committing()
        self.assertIsNone(empty_cache.plan_committed_block_drop())
        empty_cache.close()

    def test_ssm_planned_drop_includes_partial_swa_window(self) -> None:
        """Hybrid plans include SSM and every partial SWA-window page."""
        cfg = self._make_ssm_config(
            tokens_per_block=32,
            num_attn_layers=1,
            num_ssm_layers=1,
            window_size=32,
        )
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(48)]

        kv_cache = self.manager.create_kv_cache()
        kv_cache.resume(stream)
        kv_cache.capacity = len(prompt)
        kv_cache.commit(prompt)
        kv_cache.stop_committing()
        drop_handle = kv_cache.plan_committed_block_drop()
        self.assertIsNotNone(drop_handle)

        attn_lc_id = _introspection.attention_life_cycle_ids(self.manager)[0]
        ssm_lc_id = _introspection.ssm_life_cycle_id(self.manager)
        assert ssm_lc_id is not None
        num_tokens, attn_counts = _introspection.reuse_match_planned_drop_counts(
            self.manager, ReuseScope(), prompt, attn_lc_id, self.manager.enable_partial_match
        )
        self.assertEqual(num_tokens, len(prompt))
        # Every partial SWA-window attention page is planned for drop exactly once.
        self.assertTrue(attn_counts and all(count == 1 for count in attn_counts))
        _, ssm_counts = _introspection.reuse_match_planned_drop_counts(
            self.manager, ReuseScope(), prompt, ssm_lc_id, self.manager.enable_partial_match
        )
        # The SSM snapshot on the last committed block is planned for drop.
        self.assertEqual(ssm_counts[-1], 1)

        kv_cache.close()
        assert drop_handle is not None
        drop_handle.drop()
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt), 0)

    def test_ssm_same_block_snapshots_support_monotonic_multi_turn_reuse(self) -> None:
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        engine = FakeEngine(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        prompt = [self.next_token() for _ in range(64)]

        for snapshot_length, expected_reuse in ((10, 0), (20, 10), (25, 20)):
            kv_cache = self.manager.create_kv_cache(input_tokens=prompt[:snapshot_length])
            self.assertEqual(kv_cache.num_committed_tokens, expected_reuse)
            kv_cache.resume(stream)
            kv_cache.capacity = snapshot_length
            engine.execute(
                [
                    Step(
                        kv_cache,
                        prompt[expected_reuse:snapshot_length],
                        prompt[:expected_reuse],
                    )
                ],
                stream,
            )
            kv_cache.history_length = snapshot_length
            kv_cache.commit(prompt[expected_reuse:snapshot_length])
            kv_cache.close()

        exact = self.manager.create_kv_cache(input_tokens=prompt[:25])
        self.assertEqual(exact.num_committed_tokens, 25)
        exact.resume(stream)
        exact.capacity = 25
        exact.history_length = 25
        engine.execute([Step(exact, [], prompt[:25])], stream)
        exact.close()

    def test_ssm_same_block_forks_only_reuse_safe_snapshots(self) -> None:
        cfg = self._make_ssm_config(tokens_per_block=32, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        engine = FakeEngine(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(64)]

        source = self.manager.create_kv_cache()
        source.resume(stream)
        commit_start = 0
        for commit_end in (10, 20):
            source.capacity = commit_end
            chunk = prompt[commit_start:commit_end]
            engine.execute([Step(source, chunk, prompt[:commit_start])], stream)
            source.history_length = commit_end
            source.commit(chunk)
            commit_start = commit_end
        source.close()

        # The retained 20-token state is in the future of a fork at token 15.
        # Falling back to zero reuse is safe; reusing that state would corrupt
        # the fork's SSM history.
        early_fork = prompt[:15] + [self.next_token() for _ in range(25)]
        early = self.manager.create_kv_cache(input_tokens=early_fork)
        self.assertEqual(early.num_committed_tokens, 0)
        early.resume(stream)
        early.capacity = len(early_fork)
        engine.execute([Step(early, early_fork, [])], stream)
        early.history_length = len(early_fork)
        engine.execute([Step(early, [], early_fork)], stream)
        early.close()

        later_fork = prompt[:25] + [self.next_token() for _ in range(15)]
        later = self.manager.create_kv_cache(input_tokens=later_fork)
        self.assertEqual(later.num_committed_tokens, 20)
        later.resume(stream)
        later.capacity = len(later_fork)
        engine.execute([Step(later, later_fork[20:], later_fork[:20])], stream)
        later.history_length = len(later_fork)
        engine.execute([Step(later, [], later_fork)], stream)
        later.close()

        aligned = self.manager.create_kv_cache(input_tokens=prompt[:32])
        self.assertEqual(aligned.num_committed_tokens, 20)
        aligned.resume(stream)
        aligned.capacity = 32
        engine.execute([Step(aligned, prompt[20:32], prompt[:20])], stream)
        aligned.history_length = 32
        aligned.commit(prompt[20:32])
        aligned.close()

        aligned_fork = prompt[:40] + [self.next_token() for _ in range(8)]
        reused = self.manager.create_kv_cache(input_tokens=aligned_fork)
        self.assertEqual(reused.num_committed_tokens, 32)
        reused.resume(stream)
        reused.capacity = len(aligned_fork)
        engine.execute([Step(reused, aligned_fork[32:], aligned_fork[:32])], stream)
        reused.history_length = len(aligned_fork)
        engine.execute([Step(reused, [], aligned_fork)], stream)
        reused.close()

    def test_ssm_partial_snapshot_respects_partial_reuse_setting(self) -> None:
        """Partial SSM snapshots are created, but partial prompt reuse remains optional."""
        tokens_per_block = 32
        prompt = [self.next_token() for _ in range(64)]

        for enable_partial_reuse, expected_long_match in ((False, 32), (True, 48)):
            cfg = self._make_ssm_config(
                tokens_per_block=tokens_per_block,
                enable_partial_reuse=enable_partial_reuse,
            )
            self.manager = KVCacheManager(cfg)
            stream_holder = CachedCudaStream()
            stream = cast(CudaStream, stream_holder.handle)

            kv1 = self.manager.create_kv_cache()
            kv1.resume(stream)
            kv1.capacity = 32
            kv1.commit(prompt[:32])

            kv1.capacity = 48
            kv1.commit(prompt[32:48])
            kv1.close()

            longer = self.manager.create_kv_cache(input_tokens=prompt)
            self.assertEqual(longer.num_committed_tokens, expected_long_match)
            longer.resume(stream)
            longer.close()

            exact = self.manager.create_kv_cache(input_tokens=prompt[:48])
            self.assertEqual(exact.num_committed_tokens, 48)
            exact.resume(stream)
            exact.close()

            ssm_lc_id = _introspection.ssm_life_cycle_id(self.manager)
            assert ssm_lc_id is not None
            num_tokens, pages = _introspection.reuse_match_pages(
                self.manager,
                ReuseScope(),
                prompt[:48],
                ssm_lc_id,
                self.manager.enable_partial_match,
            )
            self.assertEqual(num_tokens, 48)
            last_page = pages[-1]
            assert last_page is not None
            self.assertEqual(last_page[1], 16)

            del exact, kv1, longer
            gc.collect()
            stream_holder.synchronize()
            self.manager.shutdown()
            del self.manager

    def test_commit_is_end_moves_partial_attention_and_ssm_pages(self) -> None:
        """Final partial commits move live pages into the tree instead of copying them."""
        tokens_per_block = 32
        cfg = self._make_ssm_config(tokens_per_block=tokens_per_block, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(48)]

        kv_cache = self.manager.create_kv_cache(
            expected_prompt_length=len(prompt),
            kv_cache_retention_config=KvCacheRetentionConfig(
                [
                    KvCacheRetentionConfig.TokenRangeRetentionConfig(
                        0, None, 80, timedelta(milliseconds=20)
                    )
                ]
            ),
        )
        kv_cache.resume(stream)
        kv_cache.capacity = len(prompt)
        kv_cache.history_length = len(prompt)

        attn_lc_id = _introspection.attention_life_cycle_ids(self.manager)[0]
        ssm_lc_id = _introspection.ssm_life_cycle_id(self.manager)
        assert ssm_lc_id is not None
        attn_tail_slot = kv_cache.get_base_page_indices(LayerGroupId(attn_lc_id))[1]
        ssm_slot = kv_cache.get_ssm_block_base_index(LayerGroupId(ssm_lc_id))

        kv_cache.commit(prompt, is_end=True)
        try:
            self.assertEqual(kv_cache.get_page_priorities(LayerGroupId(ssm_lc_id))[-1], 80)
        finally:
            kv_cache.close()

        _, attn_pages = _introspection.reuse_match_pages(
            self.manager, ReuseScope(), prompt, attn_lc_id, self.manager.enable_partial_match
        )
        num_tokens, ssm_pages = _introspection.reuse_match_pages(
            self.manager, ReuseScope(), prompt, ssm_lc_id, self.manager.enable_partial_match
        )
        self.assertEqual(num_tokens, len(prompt))

        attn_page = attn_pages[-1]
        ssm_page = ssm_pages[-1]
        assert attn_page is not None
        assert ssm_page is not None
        self.assertEqual(attn_page[0], attn_tail_slot)
        self.assertEqual(ssm_page[0], ssm_slot)
        self.assertEqual(ssm_page[1], 16)
        time.sleep(0.04)
        self.assertTrue(self.manager.prefetch_reuse(ReuseScope(), prompt, GPU_LEVEL))
        self.manager.refresh_retention()
        probe = self.manager.create_kv_cache(input_tokens=prompt)
        try:
            self.assertEqual(probe.get_page_priorities(LayerGroupId(ssm_lc_id))[-1], 35)
        finally:
            probe.close()

    def test_ssm_snapshot_moves_to_covering_block(self) -> None:
        """A snapshot on a partial block survives the full sibling that replaces it."""
        tokens_per_block = 32
        cfg = self._make_ssm_config(tokens_per_block=tokens_per_block, enable_partial_reuse=True)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(96)]
        ssm_lc_id = _introspection.ssm_life_cycle_id(self.manager)
        assert ssm_lc_id is not None

        # Turn 1 ends at 48 tokens, i.e. 16 tokens into block 1.
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = 48
        kv1.history_length = 48
        kv1.commit(prompt[:48])
        kv1.close()

        # Turn 2 fills blocks 0..2. Block 1 becomes a full 32-token block that replaces the
        # 16-token one, and its own SSM snapshot is taken on block 2, not block 1 -- so
        # without moving the pages over, the 48-token endpoint would be lost.
        kv2 = self.manager.create_kv_cache(input_tokens=prompt)
        kv2.resume(stream)
        kv2.capacity = len(prompt)
        kv2.history_length = len(prompt)
        kv2.commit(prompt[kv2.num_committed_tokens :])
        kv2.close()

        num_tokens, pages = _introspection.reuse_match_pages(
            self.manager, ReuseScope(), prompt[:48], ssm_lc_id, True
        )
        self.assertEqual(num_tokens, 48)
        self.assertEqual(len(pages), 2)
        self.assertIsNotNone(pages[-1])
        self.assertEqual(cast(tuple, pages[-1])[1], 16)
        # Block 1 is the full 32-token sibling, not the original 16-token block: the
        # 96-token prompt still matches end to end through it.
        full_match, _ = _introspection.reuse_match_pages(
            self.manager, ReuseScope(), prompt, ssm_lc_id, True
        )
        self.assertEqual(full_match, len(prompt))

        del kv1, kv2
        gc.collect()
        stream_holder.synchronize()

    def test_commit_min_snapshot_requires_history_alignment(self) -> None:
        """commit_min_snapshot requires commit() to start or end at history length."""
        cfg = self._make_ssm_config(tokens_per_block=32)
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(64)]

        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = 32
        kv1.commit(prompt[:32])
        kv1.close()

        kv2 = self.manager.create_kv_cache()
        kv2.resume(stream)
        kv2.capacity = 64
        kv2.history_length = 32
        kv2.commit(prompt[:32])
        kv2.close()

        kv3 = self.manager.create_kv_cache()
        kv3.resume(stream)
        kv3.capacity = 64
        kv3.history_length = 48
        with self.assertRaises(AssertionError):
            kv3.commit(prompt[:32])
        self.assertEqual(kv3.num_committed_tokens, 0)
        kv3.close()

        kv4 = self.manager.create_kv_cache()
        kv4.resume(stream)
        kv4.capacity = 48
        kv4.history_length = 48
        kv4.commit(prompt[:48])
        self.assertEqual(kv4.num_committed_tokens, 48)
        kv4.close()

    def test_ssm_reuse_config_allows_partial_reuse(self) -> None:
        config = self._make_ssm_config(enable_partial_reuse=True)
        self.assertTrue(config.enable_partial_reuse)

    def test_ssm_reuse_config_validation(self) -> None:
        """SSM reuse requires commit_min_snapshot."""
        self._make_ssm_config(enable_partial_reuse=True)
        with self.assertRaises(AssertionError):
            self._make_ssm_config(commit_min_snapshot=False)


class TestClampMaxSeqLenForMem(unittest.TestCase):
    TOKENS_PER_BLOCK = 32
    SLOT_SIZE = 2 << 20

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.managers: list[KVCacheManager] = []

    def tearDown(self) -> None:
        for manager in self.managers:
            manager.shutdown()
        gc.enable()

    def _make_manager(self, sliding_window_sizes: list[int | None]) -> KVCacheManager:
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(layer_id),
                buffers=[BufferConfig(role=Role.KEY, size=self.SLOT_SIZE)],
                sliding_window_size=window_size,
                num_sink_tokens=0 if window_size is not None else None,
            )
            for layer_id, window_size in enumerate(sliding_window_sizes)
        ]
        manager = KVCacheManager(
            KVCacheManagerConfig(
                tokens_per_block=self.TOKENS_PER_BLOCK,
                cache_tiers=[GpuCacheTierConfig(quota=len(sliding_window_sizes) * self.SLOT_SIZE)],
                layers=layers,
            )
        )
        self.managers.append(manager)
        return manager

    def test_clamp_max_seq_len_for_mem_zero_upper_bound(self):
        manager = self._make_manager([None])

        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=1, token_num_upper_bound=0), 0
        )

    def test_clamp_max_seq_len_for_mem_single_feasible_block(self):
        manager = self._make_manager([None])

        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=1, token_num_upper_bound=32), 32
        )
        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=1, token_num_upper_bound=64), 32
        )

    def test_clamp_max_seq_len_for_mem_batch_consumes_remaining_slots(self):
        manager = self._make_manager([None])

        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=2, token_num_upper_bound=64), 0
        )
        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=3, token_num_upper_bound=64), 0
        )

    def test_clamp_max_seq_len_for_mem_sliding_window_reuses_slot(self):
        manager = self._make_manager([self.TOKENS_PER_BLOCK])

        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=1, token_num_upper_bound=96), 96
        )

    def test_clamp_max_seq_len_for_mem_multiple_pool_groups(self):
        manager = self._make_manager([self.TOKENS_PER_BLOCK, None])

        # Worst-case SWA slot reservation sizes the pool to 4 slots (SWA floor 2 +
        # full-attention floor 2), so a single sequence fits the full 96 tokens.
        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=1, token_num_upper_bound=96), 96
        )
        self.assertEqual(
            manager.clamp_max_seq_len_for_mem(batch_size=2, token_num_upper_bound=96), 32
        )


class TestInitRatioConfig(unittest.TestCase):
    """Tests for init_ratio computation from typical_step and constraints."""

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()

    def tearDown(self) -> None:
        gc.enable()

    # Shared constants for all tests.
    TOKENS_PER_BLOCK = 32
    WINDOW_SIZE = 128
    SINK_TOKENS = 32
    # Non-power-of-2 sizes so granularity rounding is non-trivial.
    PG0_SLOT_SIZE = 786432  # 768KB (windowed)
    PG1_SLOT_SIZE = 1310720  # 1280KB (non-windowed)
    SSM_STATE_SLOT_SIZE = 23592960
    SSM_CONV_SLOT_SIZE = 829440
    ATTN_SLOT_SIZE = 245760

    def _make_config(
        self,
        gpu_quota: int = 128 << 20,
        typical_step: BatchDesc | None = None,
        constraints: list[BatchDesc] | None = None,
        host_quota: int = 0,
        num_windowed_layers: int = 1,
        num_full_layers: int = 1,
        enable_swa_scratch_reuse: bool = False,
        initial_pool_ratio: list[float] | None = None,
    ) -> KVCacheManagerConfig:
        """Create a config with two pool groups (windowed vs non-windowed).

        Uses large, non-power-of-2 buffer sizes so 2MB granularity rounding
        is non-trivial and constraint clamping is exercised.

        With num_windowed_layers / num_full_layers > 1 and
        scratch reuse enabled, multiple layers per lifecycle give
        frac_max < 1, making scratch savings visible in capacity planning.
        """
        cache_tiers: list = [GpuCacheTierConfig(quota=gpu_quota)]
        if host_quota > 0:
            cache_tiers.append(HostCacheTierConfig(quota=host_quota))
        layers: list = []
        lid = 0
        for _ in range(num_windowed_layers):
            layers.append(
                AttentionLayerConfig(
                    layer_id=LayerId(lid),
                    buffers=[BufferConfig(role=Role.KEY, size=self.PG0_SLOT_SIZE)],
                    sliding_window_size=self.WINDOW_SIZE,
                    num_sink_tokens=self.SINK_TOKENS,
                )
            )
            lid += 1
        for _ in range(num_full_layers):
            layers.append(
                AttentionLayerConfig(
                    layer_id=LayerId(lid),
                    buffers=[BufferConfig(role=Role.KEY, size=self.PG1_SLOT_SIZE)],
                )
            )
            lid += 1
        return KVCacheManagerConfig(
            tokens_per_block=self.TOKENS_PER_BLOCK,
            cache_tiers=cache_tiers,
            layers=layers,
            typical_step=typical_step,
            constraints=constraints or [],
            initial_pool_ratio=initial_pool_ratio,
            swa_scratch_reuse=(SwaScratchReuseConfig() if enable_swa_scratch_reuse else None),
        )

    def _make_hybrid_config(self, gpu_quota: int = 128 << 20) -> KVCacheManagerConfig:
        return KVCacheManagerConfig(
            tokens_per_block=self.TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=[
                SsmLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(
                            role=DataRole("ssm_state"),
                            size=self.SSM_STATE_SLOT_SIZE,
                        ),
                        BufferConfig(
                            role=DataRole("conv_state"),
                            size=self.SSM_CONV_SLOT_SIZE,
                        ),
                    ],
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[
                        BufferConfig(
                            role=DataRole("key"),
                            size=self.ATTN_SLOT_SIZE,
                        ),
                    ],
                ),
            ],
            enable_partial_reuse=False,
            commit_min_snapshot=True,
        )

    def test_default_init_ratio(self):
        """Without typical_step or constraints, uses hardcoded fallback."""
        cfg = self._make_config()
        manager = KVCacheManager(cfg)
        ratio = _introspection.current_gpu_ratio(manager)
        self.assertEqual(len(ratio), 2)
        self.assertAlmostEqual(sum(ratio), 1.0, places=6)
        # Windowed layers need fewer blocks than non-windowed at history=2048.
        self.assertLess(ratio[0], ratio[1])
        manager.shutdown()

    def test_typical_step_short_sequences(self):
        """typical_step with short sequences: ratio reflects buffer size difference."""
        step = BatchDesc(kv_caches=[KVCacheDesc(capacity=64, history_length=32)] * 64)
        cfg = self._make_config(typical_step=step)
        manager = KVCacheManager(cfg)
        ratio = _introspection.current_gpu_ratio(manager)
        self.assertEqual(len(ratio), 2)
        self.assertAlmostEqual(sum(ratio), 1.0, places=6)
        # Short sequences (32 tokens < window 128): no stale blocks.
        # Ratio reflects buffer size: 768KB vs 1280KB ≈ 0.6.
        self.assertAlmostEqual(ratio[0] / ratio[1], 0.6, delta=0.15)
        manager.shutdown()

    def test_typical_step_long_sequences(self):
        """typical_step with long sequences: windowed layers need less than non-windowed."""
        step = BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=4000)] * 32)
        cfg = self._make_config(typical_step=step)
        manager = KVCacheManager(cfg)
        ratio = _introspection.current_gpu_ratio(manager)
        self.assertEqual(len(ratio), 2)
        self.assertAlmostEqual(sum(ratio), 1.0, places=6)
        # Windowed layers (window=128) have many stale blocks, non-windowed keep all.
        self.assertLess(ratio[0], ratio[1])
        self.assertLess(ratio[0], 0.15)
        manager.shutdown()

    def test_zero_capacity_request_reserves_only_an_ssm_slot(self):
        """Every request reserves one SSM slot, including a zero-token dummy."""
        manager = KVCacheManager(self._make_hybrid_config())
        ssm_lc = _introspection.ssm_life_cycle_id(manager)
        assert ssm_lc is not None
        ssm_pg = manager.get_life_cycle_pool_group_indices()[ssm_lc]
        attn_pg = 1 - ssm_pg

        batch = BatchDesc(
            kv_caches=[
                KVCacheDesc(capacity=64, history_length=63),
                KVCacheDesc(capacity=0, history_length=0),
            ]
        )
        slots = _introspection.compute_slots_for_batch(manager, batch, self.TOKENS_PER_BLOCK, None)
        self.assertEqual(slots[ssm_pg], 2)
        self.assertEqual(slots[attn_pg], 2)
        manager.shutdown()

    def test_constraints_floor_typical_step(self):
        """Constraints clamp the typical_step ratio from below."""
        typical = BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=4000)] * 32)
        constraint = BatchDesc(kv_caches=[KVCacheDesc(capacity=256, history_length=128)] * 256)
        cfg_unconstrained = self._make_config(typical_step=typical)
        mgr_unconstrained = KVCacheManager(cfg_unconstrained)
        ratio_unconstrained = _introspection.current_gpu_ratio(mgr_unconstrained)

        cfg_constrained = self._make_config(typical_step=typical, constraints=[constraint])
        mgr_constrained = KVCacheManager(cfg_constrained)
        ratio_constrained = _introspection.current_gpu_ratio(mgr_constrained)

        self.assertGreater(ratio_constrained[0], ratio_unconstrained[0])
        self.assertAlmostEqual(sum(ratio_constrained), 1.0, places=6)
        mgr_unconstrained.shutdown()
        mgr_constrained.shutdown()

    def test_constraint_reserves_resume_headroom(self):
        """A full constraint batch must stay below the resume utilization gate."""
        num_requests = 32
        constraint = BatchDesc(kv_caches=[KVCacheDesc(capacity=1, history_length=0)] * num_requests)
        granularity = 2 << 20
        gpu_quota = round_up(num_requests * self.PG0_SLOT_SIZE, granularity) + round_up(
            num_requests * self.PG1_SLOT_SIZE, granularity
        )
        cfg = self._make_config(gpu_quota=gpu_quota, constraints=[constraint])
        cfg.max_util_for_resume = 0.95
        manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        kv_caches = []
        for _ in range(num_requests):
            kv_cache = manager.create_kv_cache()
            self.assertTrue(kv_cache.resume(stream))
            kv_cache.capacity = 1
            kv_caches.append(kv_cache)

        for kv_cache in kv_caches:
            kv_cache.close()
        manager.shutdown()

    def test_constraint_floor_overrides_infeasible_initial_pool_ratio(self):
        """A constraint's feasibility floor overrides an infeasible initial_pool_ratio.

        initial_pool_ratio is the target split and still overrides typical_step, but
        constraints stay feasibility floors (mirrors PR #16269): if a declared batch
        needs more slots than its target share can hold, that pool group's share is
        clamped up so the batch can be resumed, rather than starving it during warmup.
        Here pool group 1's 0.2 target cannot satisfy the 256-request constraint, so
        its share is clamped above 0.2 and pool group 0 gives up the remainder.
        """
        typical = BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=4000)] * 32)
        constraint = BatchDesc(kv_caches=[KVCacheDesc(capacity=256, history_length=128)] * 256)
        cfg = self._make_config(
            typical_step=typical,
            constraints=[constraint],
            initial_pool_ratio=[0.8, 0.2],
        )
        manager = KVCacheManager(cfg)
        ratio = _introspection.current_gpu_ratio(manager)

        self.assertGreater(ratio[1], 0.2)
        self.assertLess(ratio[0], 0.8)
        self.assertAlmostEqual(sum(ratio), 1.0, places=6)
        manager.shutdown()

    def test_initial_ratio_is_per_layer_group_when_hot_group_is_shared(self):
        config = KVCacheManagerConfig(
            tokens_per_block=self.TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=128 << 20)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role=Role.KEY, size=self.PG0_SLOT_SIZE)],
                    sliding_window_size=self.WINDOW_SIZE,
                    num_sink_tokens=self.SINK_TOKENS,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[BufferConfig(role=Role.KEY, size=self.PG0_SLOT_SIZE)],
                ),
            ],
            initial_pool_ratio=[0.25, 0.75],
        )
        manager = KVCacheManager(config)
        self.assertEqual(_introspection.current_gpu_ratio(manager), [1.0])
        manager.shutdown()

    @parameterized.expand(
        [
            ("empty", [], "initial_pool_ratio length"),
            ("wrong_length", [1.0], "initial_pool_ratio length"),
            ("zero", [0.0, 1.0], "initial_pool_ratio values must be positive"),
            ("negative", [-0.1, 1.1], "initial_pool_ratio values must be positive"),
            ("wrong_sum", [0.4, 0.5], "initial_pool_ratio values must sum to 1.0"),
        ]
    )
    def test_invalid_initial_pool_ratio(self, _name: str, ratio: list[float], error: str):
        cfg = self._make_config(initial_pool_ratio=ratio)

        with self.assertRaisesRegex(ValueError, error):
            KVCacheManager(cfg)

    @parameterized.expand(
        [("zero", 0.0), ("negative", -0.1), ("greater_than_one", 1.1), ("nan", math.nan)]
    )
    def test_invalid_max_util_for_resume(self, _name: str, max_util_for_resume: float):
        cfg = self._make_config()
        cfg.max_util_for_resume = max_util_for_resume

        with self.assertRaisesRegex((ValueError, RuntimeError), "max_util_for_resume must be in"):
            KVCacheManager(cfg)

    def test_a_slot_one_byte_under_a_grain_still_costs_a_whole_grain(self):
        """Rounding is per pool and per grain, so a slightly smaller slot buys nothing.

        Both groups cost one grain a slot -- `grain - 1` rounds up -- so five grains buy
        five slots however they are split. Which group takes the odd one is decided by the
        ratio, and these two tie exactly, so this does not pin that.
        """
        grain = 2 << 20
        cfg = KVCacheManagerConfig(
            tokens_per_block=self.TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=5 * grain)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[BufferConfig(role=Role.KEY, size=grain - 1)],
                    sliding_window_size=self.TOKENS_PER_BLOCK,
                    num_sink_tokens=0,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[BufferConfig(role=Role.KEY, size=grain)],
                ),
            ],
            constraints=[
                BatchDesc(kv_caches=[KVCacheDesc(capacity=self.TOKENS_PER_BLOCK, history_length=0)])
            ],
        )
        manager = KVCacheManager(cfg)

        def stat_slot_sizes(stat) -> list[int]:
            if hasattr(stat, "slot_sizes"):
                return stat.slot_sizes
            return stat.slot_size

        slots_by_size = {
            tuple(stat_slot_sizes(stat)): stat.total for stat in manager.get_storage_statistics()
        }
        small = slots_by_size[(grain - 1,)]
        full = slots_by_size[(grain,)]
        self.assertEqual(small + full, 5, "five grains buy five slots, not one more")
        self.assertEqual({small, full}, {2, 3}, "and they are split as evenly as five allows")
        manager.shutdown()

    @parameterized.expand([(0,), (64,), (50,), (256,)])
    def test_constraint_guarantees_batch_can_run(self, system_prompt_length: int):
        """Quota is tight; without constraint clamping the batch would fail.

        Without constraint clamping, the typical_step ratio would starve a
        pool group. With system_prompt_length > 0, a warm request commits
        the system prompt so batch requests reuse those shared blocks.
        """
        granularity = 2 << 20  # 2MB
        num_requests = 4
        capacity = 512  # > WINDOW_SIZE so windowed layers have stale blocks
        tpb = self.TOKENS_PER_BLOCK

        # sys_blocks: full blocks of the system prompt that can be shared.
        sys_blocks = system_prompt_length // tpb
        total_blocks = div_up(capacity, tpb)

        # PG1 (non-windowed): no stale blocks.
        slots_pg1 = sys_blocks + num_requests * (total_blocks - sys_blocks)

        # PG0 (windowed): stale blocks depend on history_length at resize time.
        history = system_prompt_length  # = num_committed_tokens from prefix reuse
        num_sink_blocks = self.SINK_TOKENS // tpb
        stale_beg = min(total_blocks, num_sink_blocks)
        stale_end = (
            max(stale_beg, (history + 1 - self.WINDOW_SIZE) // tpb)
            if history >= self.WINDOW_SIZE
            else stale_beg
        )
        non_stale_pg0 = total_blocks - (stale_end - stale_beg)
        stale_sys = intersect(HalfOpenRange(stale_beg, stale_end), HalfOpenRange(0, sys_blocks))
        shared_pg0 = sys_blocks - (len(stale_sys) if stale_sys else 0)
        unique_pg0 = non_stale_pg0 - shared_pg0
        slots_pg0 = shared_pg0 + num_requests * unique_pg0

        # Tight quota: exact bytes for each pool group, no padding.
        pg0_bytes = round_up(slots_pg0 * self.PG0_SLOT_SIZE, granularity)
        pg1_bytes = round_up(slots_pg1 * self.PG1_SLOT_SIZE, granularity)
        gpu_quota = pg0_bytes + pg1_bytes

        # history_length at resize time = num_committed_tokens from prefix reuse.
        resize_history = system_prompt_length
        constraint = BatchDesc(
            kv_caches=[KVCacheDesc(capacity=capacity, history_length=resize_history)]
            * num_requests,
            system_prompt_length=system_prompt_length,
        )
        typical = BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=4000)])
        cfg = self._make_config(
            gpu_quota=gpu_quota,
            typical_step=typical,
            constraints=[constraint],
            host_quota=gpu_quota,  # enables partial block copy for non-aligned sys prompts
        )
        manager = KVCacheManager(cfg)

        # Verify constraint clamping: each pool group has enough slots.
        stats = manager.get_storage_statistics()
        self.assertGreaterEqual(
            stats[0].total,
            slots_pg0,
            f"Pool group 0 must have >= {slots_pg0} slots for constraint batch",
        )
        self.assertGreaterEqual(
            stats[1].total,
            slots_pg1,
            f"Pool group 1 must have >= {slots_pg1} slots for constraint batch",
        )

        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        sys_tokens = [TokenId(i) for i in range(system_prompt_length)]

        if system_prompt_length > 0:
            # Warm request: commit system prompt into radix tree so batch reuses it.
            warm = manager.create_kv_cache(input_tokens=sys_tokens)
            warm.resume(stream)
            warm.capacity = capacity
            user_tokens = [TokenId(10000 + i) for i in range(capacity - system_prompt_length)]
            warm.commit(sys_tokens + user_tokens)
            warm.close()

        # Run the constrained batch. Without constraint clamping, resize would
        # fail with OutOfPagesError.
        kv_caches = []
        for i in range(num_requests):
            kv = manager.create_kv_cache(input_tokens=sys_tokens)
            kv.resume(stream)
            if sys_blocks > 0:
                self.assertGreaterEqual(
                    kv.num_committed_tokens,
                    sys_blocks * tpb,
                    "System prompt blocks should be reused",
                )
            kv.capacity = capacity
            kv_caches.append(kv)
        for kv in kv_caches:
            kv.close()
        manager.shutdown()

    def test_multiple_constraints_take_max(self):
        """Two constraints push different pool groups; element-wise max applies.

        c1: 8 decode requests -> needs many PG0 (windowed) slots.
        c2: 1 prefill request -> needs many PG1 (non-windowed) slots.
        Both batches must be runnable after constraint clamping.
        """
        granularity = 2 << 20
        tpb = self.TOKENS_PER_BLOCK

        c1 = BatchDesc(
            kv_caches=[KVCacheDesc(capacity=256, history_length=255)] * 8,
        )
        c2 = BatchDesc(
            kv_caches=[KVCacheDesc(capacity=2048, history_length=0)],
        )

        # Compute tight quota from the max of both constraints' PG1 needs.
        c1_pg1_slots = 8 * div_up(256, tpb)
        c2_pg1_slots = div_up(2048, tpb)
        max_pg1 = max(c1_pg1_slots, c2_pg1_slots)
        pg1_bytes = round_up(max_pg1 * self.PG1_SLOT_SIZE, granularity)
        pg0_bytes = round_up(max_pg1 * self.PG0_SLOT_SIZE, granularity)
        gpu_quota = round_up(pg0_bytes + pg1_bytes + 4 * granularity, granularity)

        cfg = self._make_config(
            gpu_quota=gpu_quota,
            constraints=[c1, c2],
            host_quota=gpu_quota,
        )
        manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        # Run c1 batch: 8 decode requests.
        kv_caches = []
        for _ in range(8):
            kv = manager.create_kv_cache()
            kv.resume(stream)
            kv.capacity = 256
            kv_caches.append(kv)
        for kv in kv_caches:
            kv.close()

        # Run c2 batch: 1 prefill request.
        kv = manager.create_kv_cache()
        kv.resume(stream)
        kv.capacity = 2048
        kv.close()

        manager.shutdown()

    def test_typical_covers_constraint_ratio_unchanged(self):
        """When typical_batch covers constraint needs, ratio is fully determined by typical_batch.

        typical_batch: 4 requests at seqLen=1024 (windowed: 5 non-stale, non-windowed: 32).
        constraint:    4 requests at seqLen=512  (windowed: 5 non-stale, non-windowed: 16).
        Since typical needs more slots in every pool group, the constraint is
        already satisfied and should not distort the ratio.
        """
        granularity = 2 << 20
        tpb = self.TOKENS_PER_BLOCK
        num_requests = 4

        typical = BatchDesc(
            kv_caches=[KVCacheDesc(capacity=1024, history_length=1024)] * num_requests,
        )
        constraint = BatchDesc(
            kv_caches=[KVCacheDesc(capacity=512, history_length=512)] * num_requests,
        )

        # Tight quota: just enough for the typical batch.
        # PG1 (non-windowed): num_requests * div_up(1024, tpb) = 4 * 32 = 128 slots
        total_blocks_pg1 = num_requests * div_up(1024, tpb)
        pg1_bytes = round_up(total_blocks_pg1 * self.PG1_SLOT_SIZE, granularity)
        pg0_bytes = round_up(total_blocks_pg1 * self.PG0_SLOT_SIZE, granularity)
        gpu_quota = pg0_bytes + pg1_bytes

        # Ratio without constraints.
        cfg_no_constraint = self._make_config(
            gpu_quota=gpu_quota,
            typical_step=typical,
        )
        mgr_no_constraint = KVCacheManager(cfg_no_constraint)
        ratio_no_constraint = _introspection.current_gpu_ratio(mgr_no_constraint)

        # Ratio with constraint that typical already covers.
        cfg_with_constraint = self._make_config(
            gpu_quota=gpu_quota,
            typical_step=typical,
            constraints=[constraint],
        )
        mgr_with_constraint = KVCacheManager(cfg_with_constraint)
        ratio_with_constraint = _introspection.current_gpu_ratio(mgr_with_constraint)

        # Ratios should be identical since typical covers the constraint.
        for i in range(len(ratio_no_constraint)):
            self.assertAlmostEqual(
                ratio_no_constraint[i],
                ratio_with_constraint[i],
                places=6,
                msg=f"PG{i} ratio changed despite typical covering constraint",
            )

        mgr_no_constraint.shutdown()
        mgr_with_constraint.shutdown()

    # ----- scratch-aware capacity planning tests -----

    def test_typical_step_scratch_reduces_windowed_ratio(self):
        """With scratch reuse, windowed PG needs fewer slots during prefill.

        16 SWA layers (frac_max=1/16) + 16 full layers.
        Typical step: 8 prefill requests (history=0, capacity=16384).

        Without scratch: both PGs need the same block count; ratio reflects
        the buffer-size difference only.
        With scratch: PG0 needs far fewer slots -> ratio shifts toward PG1.

        The quota is deliberately large and the sequences long so the SWA
        worst-case min_slots floor (sink + window blocks) is a negligible
        fraction and does not clamp the scratch-reduced windowed ratio.
        """
        step = BatchDesc(kv_caches=[KVCacheDesc(capacity=16384, history_length=0)] * 8)
        multi = dict(num_windowed_layers=16, num_full_layers=16)
        big_quota = 8 << 30
        cfg_no = self._make_config(
            gpu_quota=big_quota, typical_step=step, enable_swa_scratch_reuse=False, **multi
        )
        cfg_yes = self._make_config(
            gpu_quota=big_quota, typical_step=step, enable_swa_scratch_reuse=True, **multi
        )
        mgr_no = KVCacheManager(cfg_no)
        mgr_yes = KVCacheManager(cfg_yes)
        ratio_no = _introspection.current_gpu_ratio(mgr_no)
        ratio_yes = _introspection.current_gpu_ratio(mgr_yes)

        # With scratch: PG0 (windowed) needs far fewer slots.
        self.assertLess(ratio_yes[0], ratio_no[0])
        self.assertGreater(ratio_yes[1], ratio_no[1])

        mgr_no.shutdown()
        mgr_yes.shutdown()

    def test_constraint_with_scratch_accounts_for_scratch(self):
        """Constraint clamping uses scratch-aware slot counts.

        Tight quota computed from scratch-aware slot needs.  The batch runs
        successfully because constraint clamping allocates the right number
        of slots per pool group.
        """
        tpb = self.TOKENS_PER_BLOCK
        num_windowed = 16
        num_full = 16
        num_requests = 4
        capacity = 512
        history = 0
        granularity = 2 << 20

        total_blocks = div_up(capacity, tpb)  # 16

        # PG1 (non-windowed): no stale blocks.
        slots_pg1 = num_requests * total_blocks

        # PG0 (windowed) with scratch.
        # stale_at_capacity = [sink_blocks, (cap+1-window)//tpb)
        num_sink_blocks = div_up(self.SINK_TOKENS, tpb)  # 1
        stale_beg = min(total_blocks, num_sink_blocks)  # 1
        stale_end_at_cap = max(stale_beg, (capacity + 1 - self.WINDOW_SIZE) // tpb)  # 12
        # scratch = intersect([1,12), [0,16)) = [1,12) -> 11 blocks
        num_scratch_blocks = stale_end_at_cap - stale_beg
        # frac_max = 1/num_windowed, so scratch_slots = ceil(N / num_windowed)
        scratch_slots_per_req = div_up(num_scratch_blocks, num_windowed)  # ceil(11/16)=1
        normal_blocks = total_blocks - num_scratch_blocks  # 5
        slots_pg0 = num_requests * (normal_blocks + scratch_slots_per_req)

        # Slot sizes: num_layers_in_group * per-layer buffer size.
        pg0_slot_size = num_windowed * self.PG0_SLOT_SIZE
        pg1_slot_size = num_full * self.PG1_SLOT_SIZE

        pg0_bytes = round_up(slots_pg0 * pg0_slot_size, granularity)
        pg1_bytes = round_up(slots_pg1 * pg1_slot_size, granularity)
        gpu_quota = pg0_bytes + pg1_bytes

        constraint = BatchDesc(
            kv_caches=[KVCacheDesc(capacity=capacity, history_length=history)] * num_requests,
        )
        # typical_step: long-sequence decode (pushes ratio away from PG0).
        typical = BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=4000)])

        cfg = self._make_config(
            gpu_quota=gpu_quota,
            typical_step=typical,
            constraints=[constraint],
            enable_swa_scratch_reuse=True,
            host_quota=gpu_quota,
            num_windowed_layers=num_windowed,
            num_full_layers=num_full,
        )
        manager = KVCacheManager(cfg)

        # Verify constraint clamping: each pool group has enough slots.
        stats = manager.get_storage_statistics()
        self.assertGreaterEqual(
            stats[0].total,
            slots_pg0,
            f"Pool group 0 must have >= {slots_pg0} slots for constraint batch",
        )
        self.assertGreaterEqual(
            stats[1].total,
            slots_pg1,
            f"Pool group 1 must have >= {slots_pg1} slots for constraint batch",
        )

        # Run the constrained batch to verify it actually works.
        # With scratch reuse enabled, must use resize() instead of capacity setter.
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_caches = []
        for _ in range(num_requests):
            kv = manager.create_kv_cache()
            kv.resume(stream)
            success = kv.resize(capacity)
            self.assertTrue(success, "resize should succeed with scratch-aware constraint")
            kv_caches.append(kv)
        for kv in kv_caches:
            kv.close()
        manager.shutdown()


class TestScratchReuse(TestKVCacheManagerV2):
    """Tests for SWA prefill memory reuse (scratch slots)."""

    def _prepare_scratch(
        self,
        num_layers: int = 32,
        window_size: int = 128,
        tokens_per_block: int = 32,
        gpu_quota: int = 64 << 20,
        sink_tokens: int = 0,
        max_rewind_len: int = 0,
    ):
        """Prepare a manager with scratch reuse enabled."""
        kv_buf_size = 8192
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(i),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=kv_buf_size),
                        BufferConfig(role=DataRole("value"), size=kv_buf_size),
                    ],
                    sliding_window_size=window_size,
                    num_sink_tokens=sink_tokens,
                )
                for i in range(num_layers)
            ],
            swa_scratch_reuse=SwaScratchReuseConfig(max_rewind_len=max_rewind_len),
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

    def test_excess_scratch_slot_waits_for_ready_event_on_new_stream(self):
        num_layers = 512
        self._prepare_scratch(
            num_layers=num_layers,
            window_size=32,
            tokens_per_block=32,
            gpu_quota=16 << 20,
        )
        producer_prompt = [self.next_token() for _ in range(64)]
        consumer_prompt = [self.next_token() for _ in range(256)]
        producer = self.manager.create_kv_cache(None, producer_prompt)
        consumer = self.manager.create_kv_cache(None, consumer_prompt)
        producer_stream_holder = CachedCudaStream()
        consumer_stream_holder = CachedCudaStream()
        producer_stream = cast(CudaStream, producer_stream_holder.handle)
        consumer_stream = cast(CudaStream, consumer_stream_holder.handle)
        cached_cuda_event = get_cached_cuda_event_type()
        producer_marker = None
        # Deterministically hold the producer stream open until released from
        # the host, so the ordering assertions below cannot pass vacuously
        # just because the producer happened to finish early.
        gate = HostGate()

        try:
            self.assertTrue(producer.resume(producer_stream))
            self.assertTrue(producer.resize(64))
            with enable_kernel_delay():
                for _ in range(8):
                    self.engine.execute([Step(producer, producer_prompt, [])], producer_stream)
            gate.block_stream(producer_stream)
            producer_marker = cached_cuda_event(producer_stream)
            producer.close()

            self.assertTrue(consumer.resume(producer_stream))
            self.assertTrue(consumer.resize(256))
            self.assertTrue(consumer.has_scratch_slots)

            consumer.cuda_stream = consumer_stream
            self.assertTrue(consumer.resize(288, 256))
            self.assertFalse(consumer.has_scratch_slots)

            consumer_marker = cached_cuda_event(consumer_stream)
            # While the producer gate is held, the consumer must not be able
            # to complete: its scratch->committed migration is ordered after
            # the producer's ready event, which is gated.
            self.assertFalse(producer_marker.query_complete())
            self.assertFalse(consumer_marker.query_complete())
            gate.release()
            consumer_marker.synchronize()
            self.assertTrue(producer_marker.query_complete())
        finally:
            gate.release()
            producer_stream_holder.synchronize()
            consumer_stream_holder.synchronize()
            if producer_marker is not None and not producer_marker.is_closed():
                producer_marker.synchronize()
            if producer.status != _KVCache.Status.CLOSED:
                producer.close()
            if consumer.status != _KVCache.Status.CLOSED:
                consumer.close()
            producer_stream_holder.synchronize()
            consumer_stream_holder.synchronize()
            gate.close()

    def test_request_scratch_toggle_for_two_round_inference(self):
        self._prepare_scratch(num_layers=8, window_size=32, tokens_per_block=32, gpu_quota=16 << 20)
        prompt = [self.next_token() for _ in range(256)]
        decode_token = self.next_token()
        second_prompt = [self.next_token() for _ in range(256)]
        history: list[TokenIdExt] = []
        kv = self.manager.create_kv_cache(None, prompt)
        lg_id = LayerGroupId(0)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv.resume(stream))
            self.assertTrue(kv.resize(len(prompt)))
            self.assertTrue(kv.enable_swa_scratch_reuse)
            self.assertIsNotNone(_scratch_desc(kv, lg_id))
            self.assertTrue(kv.has_scratch_slots)
            with self.assertRaisesRegex(ValueError, "scratch blocks are needed"):
                kv.enable_swa_scratch_reuse = False

            self.engine.execute([Step(kv, prompt, history)], stream)
            kv.commit(prompt)
            history.extend(prompt)
            self.assertIsNone(_scratch_desc(kv, lg_id))
            self.assertFalse(kv.has_scratch_slots)

            kv.enable_swa_scratch_reuse = False
            self.assertFalse(kv.enable_swa_scratch_reuse)
            kv.capacity = len(history) + 1
            self.assertFalse(kv.has_scratch_slots)
            self.assertIsNone(_scratch_desc(kv, lg_id))

            self.engine.execute([Step(kv, [decode_token], history)], stream)
            kv.commit([decode_token])
            history.append(decode_token)
            self.assertFalse(kv.has_scratch_slots)

            kv.enable_swa_scratch_reuse = True
            self.assertTrue(kv.enable_swa_scratch_reuse)
            self.assertTrue(kv.resize(len(history) + len(second_prompt), len(history)))
            self.assertIsNotNone(_scratch_desc(kv, lg_id))
            self.assertTrue(kv.has_scratch_slots)

            self.engine.execute([Step(kv, second_prompt, history)], stream)
            kv.commit(second_prompt)
            history.extend(second_prompt)
            self.assertIsNone(_scratch_desc(kv, lg_id))
            self.assertFalse(kv.has_scratch_slots)
            self.engine.execute([Step(kv, [], history)], stream)
            kv.stop_committing()

        s.take_finish_event().synchronize()
        kv.close()

    def test_scratch_slot_count(self):
        """Verify peak slot count is reduced with scratch reuse.

        32 SWA layers, prompt=1024, window=128, tokens_per_block=32:
        - Without scratch: 32 coalesced slots (one per block)
        - With scratch: ceil(27/32) = 1 scratch + 5 normal = 6 slots
        """
        num_layers = 32
        window_size = 128
        tokens_per_block = 32
        prompt_len = 1024
        # Need enough GPU memory for 32 slots without scratch
        gpu_quota = 64 << 20
        self._prepare_scratch(
            num_layers=num_layers,
            window_size=window_size,
            tokens_per_block=tokens_per_block,
            gpu_quota=gpu_quota,
        )

        prompt = [self.next_token() for _ in range(prompt_len)]
        kv = self.manager.create_kv_cache(None, prompt)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = kv.resume(stream)
            self.assertTrue(success)

            # Resize to full prompt
            success = kv.resize(prompt_len)
            self.assertTrue(success)

            # Check that scratch slots are allocated
            self.assertTrue(kv.has_scratch_slots)

            # Check scratch range for the (only) layer group
            layer_groups = self.manager.layer_grouping
            self.assertEqual(len(layer_groups), 1)
            lg_id = LayerGroupId(0)

            scratch_desc = _scratch_desc(kv, lg_id)
            assert scratch_desc is not None
            self.assertIsNotNone(scratch_desc)
            num_blocks = div_up(prompt_len, tokens_per_block)  # 32

            # _get_scratch_range with hl=0, cap=1024 gives scratch = stale(1024) \ stale(0)
            num_scratch_blocks = scratch_desc.range.end - scratch_desc.range.beg
            self.assertGreater(num_scratch_blocks, 0)
            num_normal_blocks = num_blocks - num_scratch_blocks

            # num_sub_pages = num_layers (all same lifecycle) = 32
            num_sub_pages = num_layers
            expected_scratch_slots = div_up(num_scratch_blocks, num_sub_pages)
            expected_total = expected_scratch_slots + num_normal_blocks

            # Verify much less than 32 total slots
            self.assertLess(expected_total, num_blocks)

            # Scratch slot count matches expected
            self.assertEqual(len(scratch_desc.slot_ids), expected_scratch_slots)

            # Check base page indices: scratch blocks have BAD_PAGE_INDEX, normal have valid
            indices = kv.get_base_page_indices(lg_id)
            for i in range(num_blocks):
                if scratch_desc.range.beg <= i < scratch_desc.range.end:
                    self.assertEqual(
                        indices[i], BAD_PAGE_INDEX, f"Scratch block {i} should have BAD_PAGE_INDEX"
                    )
                else:
                    self.assertNotEqual(
                        indices[i], BAD_PAGE_INDEX, f"Normal block {i} has BAD_PAGE_INDEX"
                    )

            # Commit all tokens — scratch slots are released once no input blocks use scratch.
            self.engine.execute([Step(kv, prompt, [])], stream)
            kv.commit(prompt)
            kv.stop_committing()
            self.assertFalse(kv.has_scratch_slots)

        s.take_finish_event().synchronize()

        # ---------------------------------------------------------
        # Verify that scratch blocks are properly bypassed during prefix reuse.
        # 1) Exact match reuse
        prompt2 = prompt.copy()
        kv2 = self.manager.create_kv_cache(None, prompt2)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv2.resume(stream)
            kv2.resize(prompt_len)
            self.engine.execute([Step(kv2, [], prompt2)], stream)
            kv2.commit([])
            kv2.stop_committing()

        s.take_finish_event().synchronize()

        # 2) Prefix match reuse
        prompt3 = prompt[:896]
        kv3 = self.manager.create_kv_cache(None, prompt3)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv3.resume(stream)
            kv3.resize(896)
            # Since kv3 doesn't reuse out-of-window scratch blocks, input is prompt3
            self.engine.execute([Step(kv3, prompt3, [])], stream)
            kv3.commit(prompt3)
            kv3.stop_committing()

        s.take_finish_event().synchronize()

        kv.close()
        kv2.close()
        kv3.close()
        self.manager.clear_reusable_blocks()

    @parameterized.expand([(0, 7), (64, 5)])
    def test_scratch_shared_slot_ids(self, rewind_len: int, expected_scratch_blocks: int):
        """Verify that scratch blocks share coalesced slot IDs."""
        # 8 layers, window=32, tokens_per_block=32, prompt=256
        # num_sub_pages = 8 (all layers in one group)
        # rewind_len=0: blocks 0-6 are scratch, block 7 is in-window.
        # rewind_len=64: blocks 5-7 are protected from scratch by the rewind tail.
        self._prepare_scratch(
            num_layers=8,
            window_size=32,
            tokens_per_block=32,
            gpu_quota=16 << 20,
            max_rewind_len=rewind_len,
        )

        prompt = [self.next_token() for _ in range(256)]
        kv = self.manager.create_kv_cache(None, prompt)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = kv.resume(stream)
            self.assertTrue(success)

            success = kv.resize(256)
            self.assertTrue(success)

            lg_id = LayerGroupId(0)
            scratch_desc = _scratch_desc(kv, lg_id)
            assert scratch_desc is not None
            self.assertIsNotNone(scratch_desc)

            num_scratch_blocks = scratch_desc.range.end - scratch_desc.range.beg
            self.assertEqual(num_scratch_blocks, expected_scratch_blocks)

            expected_scratch_slots = div_up(expected_scratch_blocks, 8)
            self.assertEqual(len(scratch_desc.slot_ids), expected_scratch_slots)

            # Verify scratch blocks have BAD_PAGE_INDEX in base_page_indices
            indices = kv.get_base_page_indices(lg_id)
            for i in range(scratch_desc.range.beg, scratch_desc.range.end):
                self.assertEqual(
                    indices[i],
                    BAD_PAGE_INDEX,
                    f"Scratch block {i} should have BAD_PAGE_INDEX in base_page_indices",
                )

            # Verify normal block (block 7) has a valid slot_id
            self.assertNotEqual(
                indices[7], BAD_PAGE_INDEX, "Normal block should have valid slot_id"
            )

            # Verify PageIndexConverter.convert_all produces correct per-layer indices
            layer_id = LayerId(0)
            converter = self.manager.get_page_index_converter(layer_id, DataRole("key"))
            page_indices = converter(
                indices, PageIndexMode.PER_LAYER, scratch_desc.range, scratch_desc.slot_ids
            )
            # All scratch blocks should produce valid (non-BAD) page indices
            for i in range(scratch_desc.range.beg, scratch_desc.range.end):
                self.assertNotEqual(
                    page_indices[i],
                    BAD_PAGE_INDEX,
                    f"Scratch block {i} should have valid converted page index",
                )

            kv.commit(prompt)
            kv.stop_committing()

        s.take_finish_event().synchronize()
        kv.close()
        self.manager.clear_reusable_blocks()

    @parameterized.expand([(0, (0, 6), (8, 9)), (32, (0, 5), None)])
    def test_scratch_chunk_size_variation(
        self,
        rewind_len: int,
        chunk1_scratch_range: tuple[int, int],
        chunk2_scratch_range: tuple[int, int] | None,
    ):
        """Verify scratch block allocation with changing chunk sizes and multiple window sizes.

        This ensures both positive and negative net_alloc_counts code paths are tested
        simultaneously across different layers. The rewind_len parameter verifies
        that the protected rewind tail is kept out of scratch ranges.

        Layer 0: window_size = 64 (2 blocks)
        Layer 1: window_size = 256 (8 blocks)

        Chunk 1: resize(256) -> 8 blocks.
          - Layer 0 needs 6 scratch blocks without rewind, or 5 with rewind_len=32.
          - Layer 1 (stale 0-0): needs 0 scratch blocks (net_alloc_counts = 8 > 0)

        Chunk 2: resize(352, 256) -> 11 blocks.
          - Layer 0 needs 1 scratch block without rewind, or 0 with rewind_len=32.
          - Layer 1 (stale 0-3): needs 0 scratch blocks. delta_scratch = 0. New normal = 3.
            net_alloc_counts = 3 > 0
        """
        tokens_per_block = 32
        gpu_quota = 32 << 20
        kv_buf_size = 8192

        self.cfg = KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=kv_buf_size),
                        BufferConfig(role=DataRole("value"), size=kv_buf_size),
                    ],
                    sliding_window_size=64,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=kv_buf_size),
                        BufferConfig(role=DataRole("value"), size=kv_buf_size),
                    ],
                    sliding_window_size=256,
                ),
            ],
            swa_scratch_reuse=SwaScratchReuseConfig(max_rewind_len=rewind_len),
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

        prompt1 = [self.next_token() for _ in range(256)]
        prompt2 = [self.next_token() for _ in range(96)]
        kv = self.manager.create_kv_cache(None, prompt1)

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            success = kv.resume(stream)
            self.assertTrue(success)

            # Chunk 1: resize to 256
            success = kv.resize(256)
            self.assertTrue(success)

            lg_id_0 = LayerGroupId(0)
            lg_id_1 = LayerGroupId(1)

            scratch_desc_0 = _scratch_desc(kv, lg_id_0)
            self.assertIsNotNone(scratch_desc_0)
            self.assertEqual(scratch_desc_0.range.beg, chunk1_scratch_range[0])
            self.assertEqual(scratch_desc_0.range.end, chunk1_scratch_range[1])
            self.assertEqual(
                len(scratch_desc_0.slot_ids),
                chunk1_scratch_range[1] - chunk1_scratch_range[0],
            )

            # Layer 1 should have 0 scratch blocks
            scratch_desc_1 = _scratch_desc(kv, lg_id_1)
            self.assertIsNone(scratch_desc_1)

            self.engine.execute([Step(kv, prompt1, [])], stream)
            kv.commit(prompt1)

            # Test suspend/resume between chunks
            kv.suspend()
            self.assertFalse(kv.has_scratch_slots)
            success = kv.resume(stream)
            self.assertTrue(success)
            self.assertFalse(kv.has_scratch_slots)
            # Chunk 2: resize to 352 with history_length=256.
            success = kv.resize(352, 256)
            self.assertTrue(success)

            scratch_desc_0 = _scratch_desc(kv, lg_id_0)
            if chunk2_scratch_range is None:
                self.assertIsNone(scratch_desc_0)
            else:
                self.assertIsNotNone(scratch_desc_0)
                self.assertEqual(scratch_desc_0.range.beg, chunk2_scratch_range[0])
                self.assertEqual(scratch_desc_0.range.end, chunk2_scratch_range[1])

            # Layer 1 should still have 0 scratch blocks
            scratch_desc_1 = _scratch_desc(kv, lg_id_1)
            self.assertIsNone(scratch_desc_1)

            self.engine.execute([Step(kv, prompt2, prompt1)], stream)
            kv.commit(prompt2)
            kv.stop_committing()

            # Final check: verify all history
            self.engine.execute([Step(kv, [], prompt1 + prompt2)], stream)

        s.take_finish_event().synchronize()
        kv.close()
        self.manager.clear_reusable_blocks()

    def test_reuse_across_prefill_turns_keeps_only_window_minus_one(self) -> None:
        """SWA scratch reuse preserves winSize-1 history tokens across prefill turns.

        Two prefill turns, window=64, tokens_per_block=32, scratch reuse ON,
        commit_min_snapshot OFF:

        Turn 1 commits a 127-token prompt (= 2*window - 1). With scratch reuse the
        out-of-window input blocks share scratch slots, so after the turn the only
        preserved KV data is the last winSize-1 = 63 tokens (blocks 2 and 3,
        positions 64..126). Blocks 0 and 1 (positions 0..63) keep no page.

        Turn 2's prompt shares those first 127 tokens. It must still reuse the whole
        127-token committed prefix: for reuse at token 127, the first input token
        (position 127) is itself in the window, so only winSize-1 = 63 history tokens
        are required, and those are exactly the preserved blocks 2 and 3. The
        out-of-window blocks 0 and 1 are not needed even though they hold no page.
        """
        tokens_per_block = 32
        window_size = 64
        self._prepare_scratch(
            num_layers=1,
            window_size=window_size,
            tokens_per_block=tokens_per_block,
            gpu_quota=64 << 20,
        )
        swa_lc_id = _introspection.swa_life_cycle_ids(self.manager)[0]

        prompt1 = [TokenId(i) for i in range(2 * window_size - 1)]  # 127 tokens
        prompt2 = [TokenId(i) for i in range(200)]  # first 127 tokens identical to prompt1

        # ---- Turn 1: prefill + commit 127 tokens with scratch reuse, then close. ----
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv1 = self.manager.create_kv_cache()
            self.assertTrue(kv1.resume(stream))
            # resize(capacity, history_length): scratch reuse forbids the capacity setter.
            self.assertTrue(kv1.resize(len(prompt1), 0))
            self.assertTrue(kv1.has_scratch_slots)
            self.engine.execute([Step(kv1, prompt1, [])], stream)
            kv1.commit(prompt1)
            self.assertEqual(kv1.num_committed_tokens, len(prompt1))
            kv1.close()  # close() -> stop_committing() commits the partial tail block.
        s.take_finish_event().synchronize()

        # Turn 1 kept KV data only for the last winSize-1 = 63 tokens (blocks 2 and 3).
        # match() is documented volatile, so read what we need and drop the reference before
        # mutating the tree again (holding live Block refs across a later clear would leave the
        # eviction accounting inconsistent).
        num_tokens1, pages1 = _introspection.reuse_match_pages(
            self.manager, ReuseScope(), prompt1, swa_lc_id
        )
        self.assertEqual(num_tokens1, len(prompt1))
        self.assertEqual(len(pages1), 4)
        has_page = [p is not None for p in pages1]
        self.assertEqual(
            has_page,
            [False, False, True, True],  # positions 0..63 out of window; 64..126 in window
        )

        # Turn 2 reuses the full 127-token committed prefix despite blocks 0,1 lacking pages.
        self.assertEqual(self.manager.probe_reuse(input_tokens=prompt2), len(prompt1))

        # ---- Turn 2: reuse, prefill the rest, and validate the reused KV data. ----
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv2 = self.manager.create_kv_cache(input_tokens=prompt2)
            num_reused = kv2.num_committed_tokens
            self.assertEqual(num_reused, len(prompt1))
            self.assertTrue(kv2.resume(stream))
            self.assertTrue(kv2.resize(len(prompt2), num_reused))
            history = list(prompt2[:num_reused])
            inp = list(prompt2[num_reused:])
            # engine.execute checks the reused history KV against expected values, so a
            # successful run proves the reused blocks 2,3 hold correct (not scratch) data.
            self.engine.execute([Step(kv2, inp, history)], stream)
            kv2.commit(inp)
            kv2.stop_committing()
            self.engine.execute([Step(kv2, [], list(prompt2))], stream)
            kv2.close()
        s.take_finish_event().synchronize()
        self.manager.clear_reusable_blocks()


class TestPartialCoverageReuse(TestKVCacheManagerV2):
    """Pages that cover fewer tokens than their block spans.

    A rewind endpoint (a partial trailing block, e.g. 16 tokens of a 32-token block) used
    to be destroyed when a longer sibling was created, or refused when it was created
    second. The covering block may not have a committable page for every life cycle at
    that boundary. For SWA, commit_min_snapshot releases out-of-window
    pages, while scratch reuse uses temporary shared storage that is not preserved as a
    per-block page. Moving the partial sibling's pages into the covering block keeps the
    endpoint reusable, with each page tagged by its recorded token count.
    """

    TOKENS_PER_BLOCK = 32
    WINDOW_SIZE = 16

    def prepare_partial(
        self,
        gpu_quota: int = 64 << 20,
        window_size: int | None = None,
        reuse_match_backoff: int = 0,
        event_manager=None,
    ) -> None:
        kv_buf_size = 8192
        window_size = self.WINDOW_SIZE if window_size is None else window_size
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=self.TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=kv_buf_size),
                        BufferConfig(role=Role.VALUE, size=kv_buf_size),
                    ],
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=kv_buf_size),
                        BufferConfig(role=Role.VALUE, size=kv_buf_size),
                    ],
                    sliding_window_size=window_size,
                ),
            ],
            enable_partial_reuse=True,
            commit_min_snapshot=True,
            reuse_match_backoff=reuse_match_backoff,
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg, event_manager=event_manager)

    @property
    def _full_attn_lc_id(self) -> int:
        swa = set(_introspection.swa_life_cycle_ids(self.manager))
        return next(
            lc_id
            for lc_id in _introspection.attention_life_cycle_ids(self.manager)
            if lc_id not in swa
        )

    @property
    def _swa_lc_id(self) -> int:
        return _introspection.swa_life_cycle_ids(self.manager)[0]

    def run_turn(self, prompt: list[TokenIdExt], refcheck: bool = False) -> int:
        """Reuse what we can, prefill the rest, commit, close. Returns the reused count.

        `refcheck` runs FakeEngine, which validates the reused history KV against the
        expected values for every layer group, so a clean run proves the reused pages hold
        real data rather than uninitialized memory. It requires `resize()` to declare the
        pre-prefill history length so that every block the engine writes is in the SWA
        window; without it the manager is told the final length up front and skips
        allocating out-of-window blocks -- which is exactly the situation that leaves a
        partial page behind, so the two cannot be combined in the same turn.
        """
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv_cache = self.manager.create_kv_cache(input_tokens=prompt)
            num_reused = kv_cache.num_committed_tokens
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(len(prompt), num_reused if refcheck else len(prompt)))
            history = list(prompt[:num_reused])
            inp = list(prompt[num_reused:])
            if refcheck:
                self.engine.execute([Step(kv_cache, inp, history)], stream)
            kv_cache.commit(inp)
            kv_cache.close()
        s.take_finish_event().synchronize()
        return num_reused

    def _tail_coverage(self, prompt: list[TokenIdExt], lc_id: int) -> int:
        """Recorded token count of `lc_id`'s page on the last block matching `prompt`.

        Zero when that block holds no page for the life cycle. This is the tree block
        holding the tail of `prompt` (block 2 in these tests).
        """
        _, pages = _introspection.reuse_match_pages(self.manager, ReuseScope(), prompt, lc_id, True)
        self.assertTrue(pages)
        page = pages[-1]
        if page is None:
            return 0
        self.assertIsNotNone(page[1])
        return cast(int, page[1])

    def _assert_page_unlinked(self, kv_cache: Any, ordinal: int) -> None:
        """Every attention page the sequence holds at `ordinal` must be off the tree.

        A page pushed out of its slot keeps no pointer to the block, so nothing
        dereferences that block once it dies.
        """
        for lc_id in _introspection.attention_life_cycle_ids(self.manager):
            self.assertIs(
                _introspection.committed_page_is_linked(kv_cache, ordinal, lc_id),
                False,
                f"page at ordinal {ordinal} lc {lc_id} still points at a block",
            )

    def test_replaced_page_does_not_outlive_its_block(self) -> None:
        """A page pushed out of a slot must not keep a pointer to a block that dies first.

        turn1 ends inside block 2 and stays open, holding the 8-token pages it committed.
        turn2 grows that block to 16 tokens and commits over them; turn3 replaces the
        block with a 32-token one and destroys it.
        """
        self.prepare_partial()
        tpb = self.TOKENS_PER_BLOCK
        prompt = [TokenId(i) for i in range(3 * tpb)]
        turn1, turn2 = prompt[: 2 * tpb + 8], prompt[: 2 * tpb + 16]

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            kv1 = self.manager.create_kv_cache(input_tokens=turn1)
            self.assertTrue(kv1.resume(stream))
            self.assertTrue(kv1.resize(len(turn1), len(turn1)))
            kv1.commit(turn1[kv1.num_committed_tokens :], is_end=True)
            # kv1 stays open, holding block 2's 8-token pages.

            kv2 = self.manager.create_kv_cache(input_tokens=turn2)
            self.assertEqual(kv2.num_committed_tokens, len(turn1))
            self.assertTrue(kv2.resume(stream))
            self.assertTrue(kv2.resize(len(turn2), len(turn2)))
            # Block 2 becomes a 16-token block that commits over kv1's 8-token pages,
            # pushing them out of the slot while kv1 still holds them.
            kv2.commit(turn2[kv2.num_committed_tokens :], is_end=True)
            kv2.close()

            try:
                # kv1's 8-token pages left the slot but kv1 still holds them.
                self._assert_page_unlinked(kv1, 2)

                # A 32-token block replaces the 16-token one, which is then destroyed.
                self.run_turn(prompt)
                self._assert_page_unlinked(kv1, 2)
            finally:
                # Must run even on failure: leaving a sequence open makes teardown abort
                # the process, which would bury the assertion message.
                # Dereferences CommittedPage::block for the surviving 8-token pages.
                kv1.close()
        s.take_finish_event().synchronize()

    def test_replaced_reused_page_does_not_outlive_its_block(self) -> None:
        """Same defect as above, reached through reuse rather than through commit.

        The holder matches the 8-token endpoint and holds the tree's pages. It must not
        resume: resume()'s deferred copy swaps a reused partial page for a private one and
        drops the holder.
        """
        self.prepare_partial()
        tpb = self.TOKENS_PER_BLOCK
        prompt = [TokenId(i) for i in range(3 * tpb)]
        short, mid = prompt[: 2 * tpb + 8], prompt[: 2 * tpb + 16]

        self.assertEqual(self.run_turn(short), 0)

        with TemporaryCudaStream([]) as s:
            holder = self.manager.create_kv_cache(input_tokens=short)
            self.assertEqual(holder.num_committed_tokens, len(short))

            # Block 2 grows to 16 tokens, adopting the 8-token pages and then committing
            # over them while `holder` still holds them.
            try:
                self.assertEqual(self.run_turn(mid), len(short))
                self._assert_page_unlinked(holder, 2)
                # A 32-token block replaces the 16-token one, which is then destroyed.
                self.assertEqual(self.run_turn(prompt), len(mid))
                self._assert_page_unlinked(holder, 2)
            finally:
                # Must run even on failure -- see the note in the sibling test.
                holder.close()
        s.take_finish_event().synchronize()

    def test_rewind_endpoint_survives_longer_sibling_created_after(self) -> None:
        events = KVCacheEventManager(max_kv_event_entries=128)
        self.prepare_partial(event_manager=events)
        base = [TokenId(i) for i in range(80)]
        extended = base + [TokenId(i) for i in range(1000, 1080)]
        rewind = base + [TokenId(2000)]

        self.assertEqual(self.run_turn(base), 0)
        self.assertEqual(self.run_turn(extended), len(base))
        # The 16-token endpoint block is gone, but its SWA page moved into the full
        # 32-token sibling and still covers the first 16 tokens. Full coverage of
        # TOKENS_PER_BLOCK also proves the block now spans a whole block, since a page
        # never records more tokens than its block holds.
        self.assertEqual(self._tail_coverage(rewind, self._full_attn_lc_id), self.TOKENS_PER_BLOCK)
        self.assertEqual(
            self._tail_coverage(rewind, self._swa_lc_id), len(base) % self.TOKENS_PER_BLOCK
        )
        self.assertEqual(self.manager.probe_reuse(input_tokens=rewind), len(base))
        # The partial SWA page is stale at the longer endpoint and must not constrain
        # the full-attention lifecycle's reusable prefix.
        self.assertEqual(self.manager.probe_reuse(input_tokens=extended), len(extended))
        events.flush_iteration_events()
        events.get_latest_events(0)
        claimant = self.manager.create_kv_cache(
            input_tokens=rewind,
            expected_prompt_length=len(rewind),
            kv_cache_retention_config=KvCacheRetentionConfig(
                [KvCacheRetentionConfig.TokenRangeRetentionConfig(0, None, 80)]
            ),
        )
        try:
            events.flush_iteration_events()
            updates = [
                event
                for event in events.get_latest_events(0)
                if isinstance(event.data, KVCacheUpdatedData)
            ]
            self.assertTrue(updates)
            self.assertEqual({event.layer_group_id for event in updates}, {self._full_attn_lc_id})
        finally:
            claimant.close()

    def test_rewind_endpoint_attaches_to_longer_sibling_created_before(self) -> None:
        self.prepare_partial()
        base = [TokenId(i) for i in range(80)]
        extended = base + [TokenId(i) for i in range(1000, 1080)]
        rewind = base + [TokenId(2000)]

        self.assertEqual(self.run_turn(extended), 0)
        # Block 2 already spans 32 tokens, so the 80-token prompt cannot create its own
        # 16-token endpoint block; its partial pages are attached to the longer sibling.
        self.assertEqual(self.run_turn(base), 0)
        self.assertEqual(self.manager.probe_reuse(input_tokens=rewind), len(base))

    def test_reused_partial_coverage_kv_is_correct(self) -> None:
        """The salvaged endpoint must hold real KV, not uninitialized memory."""
        self.prepare_partial()
        base = [TokenId(i) for i in range(80)]
        extended = base + [TokenId(i) for i in range(1000, 1080)]
        rewind = base + [TokenId(3000 + i) for i in range(40)]

        self.run_turn(base, refcheck=True)
        self.run_turn(extended)
        # FakeEngine checks every reused history token of both layer groups, including the
        # 16 tokens of block 2 that only the salvaged partial SWA page covers.
        self.assertEqual(self.run_turn(rewind, refcheck=True), len(base))

    def test_exact_boundary_ignores_stale_last_block_partial_coverage(self) -> None:
        self.prepare_partial(window_size=1)
        base = [TokenId(i) for i in range(80)]
        boundary = base + [TokenId(i) for i in range(1000, 1016)]

        self.assertEqual(self.run_turn(base), 0)
        self.assertEqual(self.run_turn(boundary), len(base))

        self.assertEqual(
            self._tail_coverage(boundary, self._full_attn_lc_id), self.TOKENS_PER_BLOCK
        )
        self.assertEqual(self._tail_coverage(boundary, self._swa_lc_id), 16)
        # At the 96-token boundary the input token is the entire size-1 SWA window, so no
        # historical SWA block is active. The partial SWA page must not constrain the full
        # attention lifecycle's reusable prefix.
        self.assertEqual(self.manager.probe_reuse(input_tokens=boundary), len(boundary))

    def test_reuse_backoff_rechecks_swa_coverage_at_final_endpoint(self) -> None:
        self.prepare_partial(window_size=33, reuse_match_backoff=1)
        base = [TokenId(i) for i in range(48)]
        boundary = base + [TokenId(i) for i in range(1000, 1048)]

        self.assertEqual(self.run_turn(base, refcheck=True), 0)
        self.run_turn(boundary)
        self.assertEqual(self._tail_coverage(base, self._swa_lc_id), 16)

        # Backing off from 96 to 95 makes block 1's partial SWA page active, but
        # it only covers 16 of the required 32 tokens. The maximal safe endpoint
        # is the previously committed 48-token prefix.
        self.assertEqual(self.manager.probe_reuse(input_tokens=boundary), len(base))
        self.assertEqual(self.run_turn(boundary, refcheck=True), len(base))

    def test_page_coverage_only_grows(self) -> None:
        self.prepare_partial()
        base = [TokenId(i) for i in range(80)]
        longer_partial = base + [TokenId(i) for i in range(1000, 1008)]  # 88 tokens
        rewind = base + [TokenId(2000)]

        self.run_turn(base)
        self.assertEqual(self._tail_coverage(rewind, self._swa_lc_id), 16)

        self.run_turn(longer_partial)
        # A slot keeps only the widest page. The 24-token snapshot supersedes the 16-token
        # one, and it still covers the shorter rewind endpoint.
        self.assertEqual(self._tail_coverage(rewind, self._swa_lc_id), 24)
        self.assertEqual(self.manager.probe_reuse(input_tokens=rewind), len(base))
        self.assertEqual(
            self.manager.probe_reuse(input_tokens=longer_partial + [TokenId(2000)]),
            len(longer_partial),
        )

        # A later shorter snapshot cannot replace the wider page.
        self.assertEqual(self.run_turn(base), len(base))
        self.assertEqual(self._tail_coverage(rewind, self._swa_lc_id), 24)
        self.assertEqual(
            self.manager.probe_reuse(input_tokens=longer_partial + [TokenId(2000)]),
            len(longer_partial),
        )


class TestPoolRebalance(TestKVCacheManagerV2):
    """Drive the auto-tuner's pool rebalance end to end.

    Everything goes through the manager, covering
    need_adjustment -> adjust() -> adjust_cache_level -> shrink/expand_pool_group.
    """

    _TOKENS_PER_BLOCK = 32
    _PROMPT_LEN = 64
    _DECODE_LEN = 96

    def prepare_two_pool_groups(self, gpu_quota: int = 256 << 20, **rebalance_overrides) -> None:
        """Two attention life cycles with different slot sizes -> two pool groups.

        Pool groups are formed per distinct slot layout, so differing window
        sizes alone are not enough -- the buffer sizes have to differ too.

        ``rebalance_overrides`` forwards to ``PoolRebalanceConfig`` (e.g.
        ``min_sampled_kv_caches=...``) for tests that exercise non-default
        auto-tuner gating.
        """
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=self._TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=8192),
                        BufferConfig(role=Role.VALUE, size=8192),
                    ],
                    sliding_window_size=128,
                    num_sink_tokens=0,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=2048),
                        BufferConfig(role=Role.VALUE, size=2048),
                    ],
                    sliding_window_size=None,
                ),
            ],
            typical_step=BatchDesc(kv_caches=[KVCacheDesc(capacity=160, history_length=0)]),
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=160, history_length=0)])],
            pool_rebalance=PoolRebalanceConfig(**rebalance_overrides),
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

    def _gpu_ratios(self) -> list[float]:
        return list(_introspection.current_gpu_ratio(self.manager))

    def _run_sequence(
        self, prompt: list[TokenIdExt] | None = None, expect_reuse: bool = False
    ) -> list[TokenIdExt]:
        """Prefill + decode one sequence with reference checking; return its prompt.

        Passing a previously used prompt exercises block reuse, so the reference
        check reads back KV from blocks committed before the pool resize.
        """
        if prompt is None:
            prompt = [self.next_token() for _ in range(self._PROMPT_LEN)]
        kv_cache = self.manager.create_kv_cache(ReuseScope(), prompt)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv_cache.resume(stream))
            num_reused = kv_cache.num_committed_tokens
            if expect_reuse:
                self.assertGreater(
                    num_reused, 0, "replay reused no blocks; the KV check would be vacuous"
                )
            self.assertTrue(kv_cache.resize(round_up(len(prompt), self._TOKENS_PER_BLOCK)))
            capacity = kv_cache.capacity
            history = prompt[:num_reused]
            new_tokens = prompt[num_reused:]
            self.engine.execute([Step(kv_cache, new_tokens, history)], stream)
            if new_tokens:
                kv_cache.commit(new_tokens)
                history.extend(new_tokens)
            for _ in range(self._DECODE_LEN):
                if len(history) + 1 > capacity:
                    kv_cache.commit(history[kv_cache.history_length :])
                    self.assertTrue(
                        kv_cache.resize(round_up(len(history) + 1, self._TOKENS_PER_BLOCK))
                    )
                    capacity = kv_cache.capacity
                token = self.next_token()
                self.engine.execute([Step(kv_cache, [token], history)], stream)
                history.append(token)
            kv_cache.commit(history[kv_cache.history_length :])
            self.engine.execute([Step(kv_cache, [], history)], stream)
        s.take_finish_event().synchronize()
        kv_cache.close()
        return prompt

    def _slot_totals(self) -> list[int]:
        return [s.total for s in self.manager.get_storage_statistics(GPU_LEVEL)]

    def test_adjust_resizes_pool_groups(self) -> None:
        self.prepare_two_pool_groups()
        before = self._gpu_ratios()
        self.assertEqual(len(before), 2, f"expected two pool groups, got {before}")

        self._run_sequence()
        slots_before = self._slot_totals()

        # Bypass the sample-count / cooldown gates and skew the target ratio so
        # pool group 0 must grow and pool group 1 must shrink.
        _introspection.force_rebalance_precondition(self.manager, skew=2.0)
        self.assertTrue(self.manager.need_adjustment)

        self.manager.adjust()

        after = self._gpu_ratios()
        self.assertEqual(len(after), len(before))
        self.assertGreater(
            after[0] / after[1],
            before[0] / before[1],
            f"adjust() did not skew the pool ratios: {before} -> {after}",
        )
        # Guard against a no-op adjust(): the pools must really have been
        # resized, which means both the expand and the shrink path ran.
        slots_after = self._slot_totals()
        self.assertGreater(slots_after[0], slots_before[0], f"{slots_before} -> {slots_after}")
        self.assertLess(slots_after[1], slots_before[1], f"{slots_before} -> {slots_after}")

    def _clear_cooldown(self) -> None:
        """Push the last adjustment far enough back to clear the configured cooldown.

        A literal 0.0 is not enough: on a freshly booted host ``time.monotonic()``
        can be smaller than the configured cooldown, which would leave the gate shut.
        """
        cooldown = self.cfg.pool_rebalance.cooldown_secs
        _introspection.set_last_adjustment_time(self.manager, time.monotonic() - cooldown - 1.0)

    def _skew_target_ratio(self, skew: float = 2.0) -> None:
        """Perturb the target GPU ratio without touching the sample/cooldown gates."""
        current = self._gpu_ratios()
        skewed = [current[0] * skew] + current[1:]
        total = sum(skewed)
        _introspection.set_target_ratio_list_gpu(self.manager, [x / total for x in skewed])

    def test_custom_min_sampled_kv_caches_gates_adjustment(self) -> None:
        """A non-default min_sampled_kv_caches must move the sample-count gate."""
        self.prepare_two_pool_groups(min_sampled_kv_caches=5000)
        self._run_sequence()
        self._skew_target_ratio()
        self._clear_cooldown()

        # Above the default threshold (2000) but below this config's custom one.
        _introspection.set_num_sampled_kv_caches(self.manager, 2001)
        self.assertFalse(
            self.manager.need_adjustment,
            "need_adjustment fired below the configured min_sampled_kv_caches",
        )

        # Exactly at the threshold: the gate is ``<``, so this must pass.
        _introspection.set_num_sampled_kv_caches(self.manager, 5000)
        self.assertTrue(
            self.manager.need_adjustment,
            "need_adjustment did not fire once min_sampled_kv_caches was reached",
        )

    def test_custom_cooldown_secs_gates_adjustment(self) -> None:
        """A non-default cooldown_secs must move the cooldown gate."""
        self.prepare_two_pool_groups(cooldown_secs=10_000.0)
        self._run_sequence()
        self._skew_target_ratio()
        _introspection.set_num_sampled_kv_caches(self.manager, 2001)

        # Recent adjustment: still within this config's long custom cooldown.
        _introspection.set_last_adjustment_time(self.manager, time.monotonic())
        self.assertFalse(
            self.manager.need_adjustment,
            "need_adjustment fired inside the configured cooldown_secs",
        )

        # Long-past adjustment: clears even a long custom cooldown.
        self._clear_cooldown()
        self.assertTrue(
            self.manager.need_adjustment,
            "need_adjustment did not fire once cooldown_secs elapsed",
        )

    def test_custom_ratio_tolerance_gates_adjustment(self) -> None:
        """A non-default ratio_tolerance must move the deviation gate."""
        # tolerance 9.0 -> drift must exceed 10x in either direction.
        self.prepare_two_pool_groups(ratio_tolerance=9.0)
        self._run_sequence()
        _introspection.set_num_sampled_kv_caches(self.manager, 2001)
        self._clear_cooldown()

        # A 2x skew is well under this config's 10x band.
        self._skew_target_ratio(skew=2.0)
        self.assertFalse(
            self.manager.need_adjustment,
            "need_adjustment fired below the configured ratio_tolerance",
        )

        # A 20x skew clears it.
        self._skew_target_ratio(skew=20.0)
        self.assertTrue(
            self.manager.need_adjustment,
            "need_adjustment did not fire once ratio_tolerance was exceeded",
        )

    def _set_uniform_target_drift(self, drift: float) -> None:
        """Set every pool group's target so that target/current == ``drift`` exactly."""
        _introspection.set_target_ratio_list_gpu(
            self.manager, [c * drift for c in self._gpu_ratios()]
        )

    @parameterized.expand([("tolerance_below_drift", 0.5, True), ("tolerance_above", 2.0, False)])
    def test_ratio_tolerance_is_symmetric(
        self, _name: str, tolerance: float, expected: bool
    ) -> None:
        """The deviation gate must not depend on which side has drifted.

        A 2x drift and its reciprocal 0.5x are the same relative deviation, so at any
        given tolerance the auto-tuner must reach the same verdict for both.
        """
        self.prepare_two_pool_groups(ratio_tolerance=tolerance)
        self._run_sequence()
        _introspection.set_num_sampled_kv_caches(self.manager, 2001)
        self._clear_cooldown()

        for drift in (2.0, 0.5):
            with self.subTest(drift=drift):
                self._set_uniform_target_drift(drift)
                self.assertEqual(
                    self.manager.need_adjustment,
                    expected,
                    f"tolerance={tolerance} drift={drift}: expected need_adjustment={expected}",
                )

    def test_invalid_rebalance_settings_raise(self) -> None:
        """PoolRebalanceConfig must reject invalid values at construction.

        The checks live in PoolRebalanceConfig::validate() (C++), which the binding's
        constructor calls; std::invalid_argument surfaces as ValueError.
        """
        invalid_overrides = {
            "negative_min_sampled_kv_caches": dict(min_sampled_kv_caches=-1),
            "negative_cooldown_secs": dict(cooldown_secs=-1.0),
            "non_finite_cooldown_secs": dict(cooldown_secs=math.inf),
            "non_positive_target_ratio_update_interval": dict(target_ratio_update_interval=0),
            "ratio_tolerance_zero": dict(ratio_tolerance=0.0),
            "ratio_tolerance_negative": dict(ratio_tolerance=-0.25),
            "non_finite_ratio_tolerance": dict(ratio_tolerance=math.inf),
            "moving_average_decay_zero": dict(moving_average_decay=0.0),
            "moving_average_decay_one": dict(moving_average_decay=1.0),
            "non_finite_moving_average_decay": dict(moving_average_decay=math.nan),
        }
        for name, override in invalid_overrides.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    PoolRebalanceConfig(**override)

    def test_replace_preserves_pool_rebalance(self) -> None:
        """dataclasses.replace() on a config must carry pool_rebalance over.

        KVCacheManagerV2._build_cache_config rebuilds the config it just built with
        replace(config, layers=...), so anything replace() drops never reaches the
        manager. The C++ binding replaces the Python dataclass, so
        replace() is keyed on the __dataclass_fields__ spec advertised in
        tensorrt_llm/runtime/kv_cache_manager_v2/__init__.py: a field missing from
        that spec is silently reset to its constructor default.
        """
        layers = [
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(role=Role.KEY, size=1024)],
            ),
        ]
        # All exactly representable, so the round trip through the binding's float
        # ratio_tolerance and double fields compares equal.
        rebalance = PoolRebalanceConfig(
            min_sampled_kv_caches=7,
            cooldown_secs=3.5,
            target_ratio_update_interval=11,
            ratio_tolerance=0.5,
            moving_average_decay=0.5,
        )
        config = KVCacheManagerConfig(
            tokens_per_block=self._TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
            layers=layers,
            pool_rebalance=rebalance,
        )

        rebuilt = replace(config, layers=layers)

        for field, expected in (
            ("min_sampled_kv_caches", 7),
            ("cooldown_secs", 3.5),
            ("target_ratio_update_interval", 11),
            ("ratio_tolerance", 0.5),
            ("moving_average_decay", 0.5),
        ):
            with self.subTest(field):
                self.assertEqual(
                    getattr(rebuilt.pool_rebalance, field),
                    expected,
                    f"replace() dropped pool_rebalance.{field}",
                )

    def _close_one_cache(self, capacity: int, *, dummy: bool, cache_id: int) -> None:
        """Create, size and close one KV cache, optionally marked as a dummy.

        Only the close matters here: it is where the auto-tuner samples
        capacity and history length.
        """
        prompt = [self.next_token() for _ in range(self._TOKENS_PER_BLOCK)]
        kv_cache = self.manager.create_kv_cache(ReuseScope(), prompt, id=cache_id)
        if dummy:
            self.manager.mark_stats_excluded(cache_id)
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(capacity, capacity - 1))
        s.take_finish_event().synchronize()
        kv_cache.close()

    def test_dummy_kv_caches_do_not_feed_the_tuner(self) -> None:
        """Stats-excluded caches must not move the target pool ratio.

        Warmup and CUDA-graph padding requests reserve capacity at the model's
        full declared context rather than at a realistic sequence length, and
        the tuner averages the *square* of capacity -- so a handful of them
        dominates the statistic outright and the pools get sized for sequences
        that never arrive. Those caches are already marked stats-excluded at
        creation; ``_KVCache.close`` has to honour that.
        """
        self.prepare_two_pool_groups()

        # Open the sample-count and cooldown gates but leave the target ratio
        # alone -- unlike force_rebalance_precondition, which also skews it.
        # need_adjustment is then driven purely by whether a close moved the
        # target away from the current ratio.
        _introspection.set_num_sampled_kv_caches(self.manager, 2001)
        _introspection.set_last_adjustment_time(self.manager, 0.0)
        self.assertFalse(
            self.manager.need_adjustment,
            "target should still equal the current ratio before any close",
        )

        big = 64 * self._TOKENS_PER_BLOCK
        self._close_one_cache(big, dummy=True, cache_id=1)
        self.assertFalse(
            self.manager.need_adjustment,
            "a stats-excluded cache moved the target ratio; dummy requests are "
            "feeding the auto-tuner",
        )

        # Control: the same close *without* the exclusion must move the target,
        # otherwise the assertion above passes for the wrong reason.
        self._close_one_cache(big, dummy=False, cache_id=2)
        self.assertTrue(
            self.manager.need_adjustment,
            "a real close of the same shape did not move the target either, so "
            "the check above is vacuous",
        )

    def test_kv_survives_adjust(self) -> None:
        """Committed blocks must still verify after pages migrate between slots."""
        self.prepare_two_pool_groups()
        self.assertEqual(len(self._gpu_ratios()), 2)

        prompt = self._run_sequence()

        _introspection.force_rebalance_precondition(self.manager, skew=2.0)
        self.manager.adjust()

        # Replay the same prompt: it reuses the committed blocks, and the fake
        # engine's reference check reads back the KV those blocks point at.
        self._run_sequence(prompt=prompt, expect_reuse=True)


class TestCachedTokensByTier(TestKVCacheManagerV2):
    @contextmanager
    def _tiered_prefix(self):
        """Three committed blocks of 4 tokens spread over the configured cache levels.

        Per-(block, life cycle) levels are (0,0), (1,0), (1,2), so a match of 11 tokens makes
        blocks 0 and 1 full reuses and block 2 a partial one, with the two attention life cycles
        disagreeing on where the pages sit. Yields (tokens, life_cycles).
        """
        self.prepare(16 << 20, 16 << 20, 16 << 20, 2, 16, 0, tokens_per_block=4)
        tokens = [TokenId(i) for i in range(12)]
        life_cycles = _introspection.attention_life_cycle_ids(self.manager)
        coverage = [4 if i in life_cycles else 0 for i in range(max(life_cycles) + 1)]
        blocks = []
        try:
            parent = None
            for ordinal, levels in enumerate(((0, 0), (1, 0), (1, 2))):
                block = _introspection.make_test_block(
                    self.manager, tokens[ordinal * 4 : (ordinal + 1) * 4], coverage, parent
                )
                blocks.append(block)
                for life_cycle, level in zip(life_cycles, levels):
                    if level:
                        _introspection.set_test_block_page_cache_level(
                            block, life_cycle, CacheLevel(level)
                        )
                parent = block
            yield tokens, life_cycles
        finally:
            # Put every page back on GPU before closing, so teardown does not trip over a block
            # whose pages were parked on a cold level.
            for block in blocks:
                for life_cycle in life_cycles:
                    _introspection.set_test_block_page_cache_level(block, life_cycle, CacheLevel(0))
            for block in reversed(blocks):
                _introspection.close_test_block(block)

    def test_current_residency(self) -> None:
        with self._tiered_prefix() as (tokens, _):
            cache = self.manager.create_kv_cache(input_tokens=tokens[:11])
            try:
                # Levels 0/1/2 are gpu/host/disk in this fixture's configured tier list.
                self.assertEqual(list(cache.cached_tokens_by_level), [4, 4, 3])
                self.assertEqual(cache._get_last_cached_token_level(), 2)
            finally:
                cache.close()

    def test_reused_blocks_by_level(self) -> None:
        """Reuse block counts are split by the cache level each reused page sat on."""
        with self._tiered_prefix() as (tokens, life_cycles):
            cache = self.manager.create_kv_cache(input_tokens=tokens[:11])
            try:
                cache.commit_pending_stats()
                by_level = self.manager.get_and_reset_iteration_reused_blocks_by_level()

                # First life cycle: block 0 on level 0, block 1 on level 1 (both full), block 2
                # on level 1 (partial). Second: blocks 0 and 1 on level 0, block 2 on level 2.
                self.assertEqual(list(by_level[life_cycles[0]].full), [1, 1, 0])
                self.assertEqual(list(by_level[life_cycles[0]].partial), [0, 1, 0])
                self.assertEqual(list(by_level[life_cycles[1]].full), [2, 0, 0])
                self.assertEqual(list(by_level[life_cycles[1]].partial), [0, 0, 1])
                # Draining resets the accumulator.
                self.assertEqual(self.manager.get_and_reset_iteration_reused_blocks_by_level(), {})
            finally:
                cache.close()

    def test_iteration_accumulation(self) -> None:
        """Attribution is staged with the reuse match and committed with the sequence's stats."""
        with self._tiered_prefix() as (tokens, _):
            cache = self.manager.create_kv_cache(input_tokens=tokens[:11])
            try:
                # Nothing is reported until the sequence's pending stats are committed.
                self.assertEqual(
                    list(self.manager.get_and_reset_iteration_cached_tokens_by_level()), []
                )
                cache.commit_pending_stats()
                self.assertEqual(
                    list(self.manager.get_and_reset_iteration_cached_tokens_by_level()), [4, 4, 3]
                )
                # Draining resets the accumulator, and committing again does not re-report the
                # same match.
                cache.commit_pending_stats()
                self.assertEqual(
                    list(self.manager.get_and_reset_iteration_cached_tokens_by_level()), []
                )
            finally:
                cache.close()

    def test_drop_cached_token_attribution(self) -> None:
        """Dropping suppresses the whole match, e.g. for a KV cache sizing dry run."""
        with self._tiered_prefix() as (tokens, _):
            cache = self.manager.create_kv_cache(input_tokens=tokens[:11])
            try:
                cache.drop_cached_token_attribution()
                cache.commit_pending_stats()
                self.assertEqual(
                    list(self.manager.get_and_reset_iteration_cached_tokens_by_level()), []
                )
            finally:
                cache.close()

    def test_drop_partial_block_cached_token_attribution(self) -> None:
        """Only the trailing partial block stops counting; complete blocks survive."""
        with self._tiered_prefix() as (tokens, _):
            # 11 tokens is two full blocks plus a 3-token tail resident on level 2.
            cache = self.manager.create_kv_cache(input_tokens=tokens[:11])
            try:
                cache.drop_partial_block_cached_token_attribution()
                cache.commit_pending_stats()
                self.assertEqual(
                    list(self.manager.get_and_reset_iteration_cached_tokens_by_level()), [4, 4, 0]
                )
            finally:
                cache.close()

    def test_drop_partial_block_keeps_block_aligned_attribution(self) -> None:
        with self._tiered_prefix() as (tokens, _):
            cache = self.manager.create_kv_cache(input_tokens=tokens[:8])
            try:
                cache.drop_partial_block_cached_token_attribution()
                cache.commit_pending_stats()
                self.assertEqual(
                    list(self.manager.get_and_reset_iteration_cached_tokens_by_level()), [4, 4, 0]
                )
            finally:
                cache.close()


@pytest.mark.cpu_only
class TestBlockKeyHashing(unittest.TestCase):
    """Verify blockchain keys are bit-identical to the per-token reference (no GPU needed)."""

    @staticmethod
    def _chain(block: "list[int | bytes]") -> bytes:
        """The single block key for one whole-sequence block, and its parent root key."""
        keys = list(sequence_to_blockchain_keys(max(len(block), 1), ReuseScope(), block))
        return keys[-1][1]

    @staticmethod
    def _root() -> bytes:
        return next(iter(sequence_to_blockchain_keys(1, ReuseScope(), [])))[1]

    @staticmethod
    def _ref_update(parent_key: bytes, block: "list[int | bytes]") -> bytes:
        h = hashlib.sha256()
        h.update(parent_key)
        for item in block:
            # Normal token ids are packed as 4 little-endian bytes (31-bit range),
            # matching the 4-byte TokenIdExt layout.
            h.update(item.to_bytes(4, "little") if type(item) is int else item)
        return h.digest()

    def test_block_key_int_block_matches_reference(self) -> None:
        rng = random.Random(123)
        parent_key = self._root()
        for n in (1, 7, 32, 33, 257):
            block = [rng.randint(0, (1 << 31) - 1) for _ in range(n)]
            self.assertEqual(
                self._chain(block),
                self._ref_update(parent_key, block),
                f"int block of length {n}",
            )

    def test_block_keys_chain_across_blocks(self) -> None:
        """Each block key hashes its parent's key, so block N depends on blocks 0..N-1."""
        rng = random.Random(4242)
        tokens_per_block = 32
        # Three whole blocks plus a short tail: the tail is chunked like any other block,
        # so it gets a key of its own.
        tokens = [rng.randint(0, (1 << 31) - 1) for _ in range(3 * tokens_per_block + 5)]

        keys = [
            key for _, key in sequence_to_blockchain_keys(tokens_per_block, ReuseScope(), tokens)
        ]
        self.assertEqual(keys[0], self._root())
        self.assertEqual(len(keys), 5)

        expected = self._root()
        for ordinal in range(4):
            block = tokens[ordinal * tokens_per_block : (ordinal + 1) * tokens_per_block]
            expected = self._ref_update(expected, block)
            self.assertEqual(keys[ordinal + 1], expected, f"block {ordinal}")

        # Chaining runs forward only: rewriting the tail must not disturb the keys of the
        # blocks before it, which is what makes a matched prefix reusable.
        tail_rewritten = tokens[:-1] + [tokens[-1] ^ 1]
        rewritten_keys = [
            key
            for _, key in sequence_to_blockchain_keys(
                tokens_per_block, ReuseScope(), tail_rewritten
            )
        ]
        self.assertEqual(rewritten_keys[:-1], keys[:-1])
        self.assertNotEqual(rewritten_keys[-1], keys[-1])

    def test_block_key_mixed_multimodal_block(self) -> None:
        block = [randbytes(32), 5, 6, randbytes(32)] + list(range(20))
        self.assertEqual(self._chain(block), self._ref_update(self._root(), block))

    def test_multimodal_digest_requires_sha256_length(self) -> None:
        for digest_size in (31, 33):
            with self.subTest(digest_size=digest_size), self.assertRaises(ValueError):
                gen_multimodal_cache_key_tokens(100, bytes(digest_size), 1)


class TestPoison(TestKVCacheManagerV2):
    """The refusal path taken once KVCM2 records a broken invariant.

    Poisoning is set directly rather than by provoking a real violation, so these exercise the
    refusal and recovery machinery without depending on a specific bug.

    Each test disposes of its manager itself: leaving one alive would keep the latch
    un-clearable, and every test after it would then be blamed for this test's poisoning.
    """

    def setUp(self) -> None:
        super().setUp()
        if not hasattr(_introspection, "poison_for_testing"):
            raise unittest.SkipTest("the poison latch is C++-backend only")

    def _discard_poisoned_manager(self) -> None:
        del self.manager
        gc.collect()
        self.assertEqual(num_live_managers(), 0)

    def test_poisoned_manager_rejects_api_calls(self) -> None:
        self.prepare(5619712, 0, 0, 8, None, 0)
        # Held open across the poisoning. Disposal is exempt in two ways, and an open cache is
        # what separates them: shutdown() must return rather than raise CorruptedError, and it
        # must skip its cleanup, whose _checkNoLivingKvCaches() this cache would otherwise trip.
        kv_cache = self.manager.create_kv_cache()
        _introspection.poison_for_testing("deliberate test poisoning")

        # The refusal names the original violation, not whatever inconsistency the refused call
        # would have tripped over.
        with self.assertRaisesRegex(CorruptedError, "deliberate test poisoning"):
            self.manager.create_kv_cache()

        # A caller that detects corruption still has to be able to tear the manager down.
        self.manager.shutdown()

        del kv_cache
        self._discard_poisoned_manager()
        self.assertIn("deliberate test poisoning", take_poison())

    def test_take_poison_only_clears_once_no_manager_is_alive(self) -> None:
        self.prepare(5619712, 0, 0, 8, None, 0)
        # Two managers, so that "every manager is gone" is distinguishable from "a manager is
        # gone": with only one, an implementation that cleared on any destruction would pass.
        second = KVCacheManager(self.cfg)
        _introspection.poison_for_testing("deliberate test poisoning")

        # A live manager skipped its own cleanup when it was poisoned, so clearing now would
        # resume work on structures known to be inconsistent. The reason is still reported.
        self.assertEqual(num_live_managers(), 2)
        self.assertIn("deliberate test poisoning", take_poison())
        self.assertIsNotNone(poison_reason())

        del second
        gc.collect()
        self.assertEqual(num_live_managers(), 1)
        self.assertIn("deliberate test poisoning", take_poison())
        self.assertIsNotNone(poison_reason())

        self._discard_poisoned_manager()

        self.assertIn("deliberate test poisoning", take_poison())
        self.assertIsNone(poison_reason())


if __name__ == "__main__":
    unittest.main()
