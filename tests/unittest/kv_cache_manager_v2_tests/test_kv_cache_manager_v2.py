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
import functools
import gc
import hashlib
import itertools
import math
import os
import random
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from random import randbytes
from statistics import median
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Sequence, cast, get_type_hints

import pytest

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        DEFAULT_BEAM_INDEX,
        AttentionLayerConfig,
        BatchDesc,
        BufferConfig,
        BufferId,
        CacheLevel,
        CudaStream,
        DataRole,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheDesc,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerGroupId,
        LayerId,
        PlannedDropHandle,
        ReuseScope,
        SsmLayerConfig,
        SwaScratchReuseConfig,
        TokenId,
        TokenIdExt,
        _introspection,
        _KVCache,
        gen_multimodal_cache_key_tokens,
    )
    from kv_cache_manager_v2._block_radix_tree import Hasher
    from kv_cache_manager_v2._common import (
        BAD_PAGE_INDEX,
        GPU_LEVEL,
        CacheTier,
        MemAddress,
        PageIndexMode,
        SlidingWindowSize,
    )
    from kv_cache_manager_v2._copy_engine import CopyTask, batched_copy
    from kv_cache_manager_v2._exceptions import LogicError, OutOfPagesError
    from kv_cache_manager_v2._storage._core import CacheLevelStorage, PoolGroupBase, SlotAllocator
    from kv_cache_manager_v2._storage_manager import StorageManager
    from kv_cache_manager_v2._utils import (
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
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        DEFAULT_BEAM_INDEX,
        AttentionLayerConfig,
        BatchDesc,
        BufferConfig,
        BufferId,
        CacheLevel,
        CudaStream,
        DataRole,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheDesc,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerGroupId,
        LayerId,
        PlannedDropHandle,
        ReuseScope,
        SsmLayerConfig,
        SwaScratchReuseConfig,
        TokenId,
        TokenIdExt,
        _introspection,
        _KVCache,
        gen_multimodal_cache_key_tokens,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import Hasher
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
        BAD_PAGE_INDEX,
        GPU_LEVEL,
        CacheTier,
        MemAddress,
        PageIndexMode,
        SlidingWindowSize,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import CopyTask, batched_copy
    from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import LogicError, OutOfPagesError
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage._core import (
        CacheLevelStorage,
        PoolGroupBase,
        SlotAllocator,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import StorageManager
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
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

from copy import deepcopy

from parameterized import parameterized

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from fake_engine import FakeEngine, Role, Step
    from kernels import HostGate, enable_kernel_delay


KV_CACHE_MANAGER_V2_BACKEND = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower()

# Gate for white-box tests that reach into the pure-Python implementation's objects
# (e.g. mutating a CommittedPage field). Prefer `_introspection`, which works on both
# backends; use this only when the behaviour under test cannot be reached through it.
requires_python_backend = unittest.skipIf(
    KV_CACHE_MANAGER_V2_BACKEND == "cpp",
    "white-box test over pure-Python KVCacheManagerV2 internals",
)

requires_cpp_backend = unittest.skipUnless(
    KV_CACHE_MANAGER_V2_BACKEND == "cpp",
    "cold-page codec end-to-end test requires the C++ backend",
)


def get_cached_cuda_event_type():
    backend = KV_CACHE_MANAGER_V2_BACKEND
    if backend == "cpp":
        try:
            from bindings.internal.batch_manager.kv_cache_manager_v2 import CachedCudaEvent

            return CachedCudaEvent
        except ImportError:
            from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2 import (
                CachedCudaEvent,
            )

            return CachedCudaEvent

    if find_spec("kv_cache_manager_v2") is not None:
        from kv_cache_manager_v2._utils import CachedCudaEvent

        return CachedCudaEvent
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import CachedCudaEvent

    return CachedCudaEvent


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


class TestTypedSlotIds(unittest.TestCase):
    def test_num_slots_accessors_return_int(self) -> None:
        self.assertIs(get_type_hints(SlotAllocator.num_slots.fget)["return"], int)
        self.assertIs(get_type_hints(SlotAllocator.num_free_slots.fget)["return"], int)
        self.assertIs(get_type_hints(SlotAllocator.num_occupied_slots.fget)["return"], int)
        self.assertIs(get_type_hints(PoolGroupBase.num_slots.fget)["return"], int)
        self.assertIs(get_type_hints(PoolGroupBase.num_free_slots.fget)["return"], int)
        self.assertIs(get_type_hints(CacheLevelStorage.num_slots)["return"], int)
        self.assertIs(get_type_hints(CacheLevelStorage.get_num_free_slots)["return"], int)
        self.assertIs(get_type_hints(StorageManager.num_slots)["return"], int)

        self.assertIs(get_type_hints(SlotAllocator.allocate_multiple)["num_slots"], int)
        self.assertIs(get_type_hints(PoolGroupBase.allocate_multiple)["num_slots"], int)
        self.assertIs(get_type_hints(CacheLevelStorage.allocate_multiple)["num_slots"], int)
        self.assertIs(get_type_hints(StorageManager.new_slots_for_pool_group)["num_slots"], int)

        allocator = SlotAllocator(3)
        self.assertEqual(allocator.num_slots, 3)
        self.assertEqual(allocator.num_free_slots, 3)
        self.assertEqual(allocator.num_occupied_slots, 0)


class TestCacheLevelStorage(unittest.TestCase):
    def test_grains_to_slots_refines_proportional_lower_bound(self) -> None:
        granularity = 16 << 20
        slot_size_list = [16_252_928, 4_063_232]
        min_slots = 157

        grains = _introspection.grains_for_slots(min_slots, slot_size_list, granularity)
        slots, used = _introspection.grains_to_slots(grains, slot_size_list, granularity)

        self.assertGreaterEqual(slots, min_slots)
        self.assertLessEqual(used, grains)

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
            self.manager.shutdown()
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


class TestNoBatching(TestKVCacheManagerV2):
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
        # workaround a mypyc bug: exception in property setter is not propagated
        # kv_cache.capacity = round_up(len(prompt), interval)
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
                # workaround a mypyc bug: exception in property setter is not propagated
                # kv_cache.capacity = round_up(required_capacity, interval)
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

    @requires_cpp_backend
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
        hot_groups = [
            _introspection.pool_group_index(self.manager, lc_id, 0)
            for lc_id in (full_lc, short_swa_lc, long_swa_lc)
        ]
        cold_groups = [
            _introspection.pool_group_index(self.manager, lc_id, 1)
            for lc_id in (full_lc, short_swa_lc, long_swa_lc)
        ]
        self.assertNotEqual(hot_groups[0], hot_groups[1])
        self.assertEqual(hot_groups[1], hot_groups[2])
        self.assertEqual(cold_groups[0], cold_groups[1])
        self.assertNotEqual(cold_groups[1], cold_groups[2])

        hot_stats = _introspection.storage_statistics(self.manager, 0)
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

    @requires_cpp_backend
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
        hot_groups = [
            _introspection.pool_group_index(self.manager, lc_id, 0) for lc_id in (full_lc, swa_lc)
        ]
        cold_groups = [
            _introspection.pool_group_index(self.manager, lc_id, 1) for lc_id in (full_lc, swa_lc)
        ]
        self.assertEqual(hot_groups[0], hot_groups[1])
        self.assertNotEqual(cold_groups[0], cold_groups[1])

        hot_stats = _introspection.storage_statistics(self.manager, 0)
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
            for stat in _introspection.storage_statistics(self.manager):
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

    @requires_cpp_backend
    def test_resume_ignores_threshold_for_ssm_only_pool_group(self) -> None:
        """A saturated SSM-only pool group must not veto resume.

        max_util_for_resume reserves room for admitted requests to grow, but an
        SSM state is a fixed one slot per sequence and never grows. Gating on it
        would deadlock a hybrid model at high concurrency, since that pool sits
        at ~100% whenever it is full.
        """
        cfg = KVCacheManagerConfig(
            tokens_per_block=32,
            cache_tiers=[GpuCacheTierConfig(quota=512 << 20)],
            max_util_for_resume=0.9,
            layers=[
                SsmLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=DataRole("ssm_state"), size=23592960),
                        BufferConfig(role=DataRole("conv_state"), size=829440),
                    ],
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[BufferConfig(role=DataRole("key"), size=245760)],
                ),
            ],
            enable_partial_reuse=False,
            commit_min_snapshot=True,
        )
        self.manager = KVCacheManager(cfg)

        # The SSM lifecycle must own a pool group containing no attention
        # lifecycle, otherwise the headroom legitimately applies to it and this
        # test would be vacuous. Assert that rather than assume it.
        ssm_lc = _introspection.ssm_life_cycle_id(self.manager)
        self.assertIsNotNone(ssm_lc)
        ssm_pg = _introspection.pool_group_index(self.manager, ssm_lc)
        attn_pgs = {
            _introspection.pool_group_index(self.manager, lc)
            for lc in _introspection.attention_life_cycle_ids(self.manager)
        }
        self.assertNotIn(ssm_pg, attn_pgs)
        self.assertEqual(len(attn_pgs), 1)
        attn_pg = next(iter(attn_pgs))

        def utilization() -> list[float]:
            return list(_introspection.storage_utilization(self.manager, GPU_LEVEL))

        def ssm_free() -> int:
            return _introspection.storage_statistics(self.manager)[ssm_pg].free

        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prior_caches: list[_KVCache] = []
        try:
            # Consume SSM slots (one per sequence) until that pool group is past
            # the resume threshold but still has a slot left to hand out.
            for _ in range(64):
                if utilization()[ssm_pg] > cfg.max_util_for_resume and ssm_free() > 0:
                    break
                kv_cache = self.manager.create_kv_cache()
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(cfg.tokens_per_block))
                prior_caches.append(kv_cache)

            utilizations = utilization()
            self.assertGreater(utilizations[ssm_pg], cfg.max_util_for_resume)
            self.assertLess(utilizations[attn_pg], cfg.max_util_for_resume)
            self.assertGreater(ssm_free(), 0)

            # Previously the gate compared max() across pool groups against a
            # single scalar, so the saturated SSM group rejected this resume.
            admitted = self.manager.create_kv_cache()
            prior_caches.append(admitted)
            self.assertTrue(admitted.resume(stream))
            self.assertTrue(admitted.resize(cfg.tokens_per_block))
        finally:
            for kv_cache in prior_caches:
                if kv_cache.status != _KVCache.Status.CLOSED:
                    kv_cache.close()

    @requires_cpp_backend
    def test_constant_size_pool_group_floor_ignores_growth_headroom(self) -> None:
        """An SSM-only pool group is sized to its exact constraint floor.

        Constraint-derived floors are inflated by 1/max_util_for_resume so an
        admitted sequence has room to grow. A pool group whose life cycles all
        have a constant per-sequence state size never grows and is never gated
        on that threshold, so inflating its floor only wastes memory.
        """
        ssm_floor_slots = 12

        def ssm_pool_slots(max_util: float, with_constraint: bool) -> int:
            # Zero-capacity requests cost no attention pages but reserve one
            # SSM slot each, so they isolate the recurrent floor.
            constraints = (
                [BatchDesc([KVCacheDesc(capacity=0, history_length=0)] * ssm_floor_slots)]
                if with_constraint
                else []
            )
            cfg = KVCacheManagerConfig(
                tokens_per_block=32,
                cache_tiers=[GpuCacheTierConfig(quota=256 << 20)],
                max_util_for_resume=max_util,
                layers=[
                    SsmLayerConfig(
                        layer_id=LayerId(0),
                        buffers=[
                            BufferConfig(role=DataRole("ssm_state"), size=23592960),
                            BufferConfig(role=DataRole("conv_state"), size=829440),
                        ],
                    ),
                    AttentionLayerConfig(
                        layer_id=LayerId(1),
                        buffers=[BufferConfig(role=DataRole("key"), size=245760)],
                    ),
                ],
                constraints=constraints,
                enable_partial_reuse=False,
                commit_min_snapshot=True,
            )
            manager = KVCacheManager(cfg)
            try:
                ssm_lc = _introspection.ssm_life_cycle_id(manager)
                self.assertIsNotNone(ssm_lc)
                ssm_pg = _introspection.pool_group_index(manager, ssm_lc)
                return _introspection.storage_statistics(manager)[ssm_pg].total
            finally:
                manager.shutdown()

        # Precondition: the floor must actually bind. If ratio-based sizing
        # already exceeded it, the assertions below would hold vacuously.
        self.assertLess(ssm_pool_slots(1.0, with_constraint=False), ssm_floor_slots)

        # The floor is honoured exactly and does not scale with the headroom
        # factor. Previously max_util=0.5 doubled it to 24 slots.
        for max_util in (1.0, 0.5):
            self.assertEqual(ssm_pool_slots(max_util, with_constraint=True), ssm_floor_slots)

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

    def test_planned_drop_handle(self) -> None:
        window_size = 8
        self.prepare(16 << 20, 0, 0, 2, window_size, 0, tokens_per_block=8)
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
        self.assertEqual(self.manager.probe_reuse(None, short_tokens), len(short_tokens))

        short_handle.drop()
        self.assertEqual(self.manager.probe_reuse(None, short_tokens), 0)
        self.assertEqual(self.manager.probe_reuse(None, long_tokens), len(long_tokens))

        long_handle.drop()
        # The SWA window is dropped, while older full-attention blocks remain reusable.
        self.assertEqual(
            self.manager.probe_reuse(None, long_tokens), len(long_tokens) - window_size
        )
        with self.assertRaisesRegex(ValueError, "already been dropped"):
            long_handle.drop()

    @requires_python_backend
    def test_planned_drop_handle_rejects_partial_coverage(self) -> None:
        # plan_committed_block_drop() rejects via _prune_match, which clamps the match to
        # the page's recorded token count, so the endpoint no longer matches exactly.
        # Forcing that state needs a direct write to the page, hence the backend gate.
        window_size = 8
        tokens_per_block = 8
        self.prepare(16 << 20, 0, 0, 2, window_size, 0, tokens_per_block=tokens_per_block)
        tokens = [self.next_token() for _ in range(3 * tokens_per_block)]

        with TemporaryCudaStream([]) as stream_holder:
            stream = cast(CudaStream, stream_holder.handle)
            kv_cache = self.manager.create_kv_cache(None, tokens)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(len(tokens)))
            kv_cache.commit(tokens)
            kv_cache.stop_committing()

            swa_lc_id = next(
                lc_id
                for lc_id, lc in self.manager._life_cycles.attention_life_cycles()
                if lc.window_size is not None
            )
            tree_block = kv_cache._blocks[2].tree_block
            assert tree_block is not None
            page = tree_block.get_page(swa_lc_id)
            assert page is not None
            self.assertEqual(page.num_tokens_in_block, len(tree_block.tokens))

            page.num_tokens_in_block -= 1
            try:
                self.assertIsNone(kv_cache.plan_committed_block_drop())
            finally:
                page.num_tokens_in_block += 1
                kv_cache.close()
        stream_holder.take_finish_event().synchronize()

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
        # The int32-ndarray ingest fast path lives in the C++ binding; the pure-Python
        # backend consumes plain lists (the dispatcher hands it get_tokens, not a view).
        if os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower() == "python":
            self.skipTest("int32-ndarray ingest is a C++-backend fast path")

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
        scoped = ReuseScope(lora_id=7, salt=11)

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
        self.assertEqual(num_reused(ReuseScope(lora_id=7, salt=12)), 0)
        self.assertEqual(num_reused(ReuseScope(lora_id=8, salt=11)), 0)
        self.assertEqual(num_reused(ReuseScope(lora_id=7)), 0)
        self.assertEqual(num_reused(ReuseScope(salt=11)), 0)
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
    detached subtree, which silently discards work on the Python backend and segfaults on
    the C++ one, so both entry points must reject the call instead.
    """

    # The two backends raise different types: the Python backend raises its own
    # LogicError, while the C++ backend's TLLM_CHECK_WITH_INFO throws a TllmException
    # (a std::runtime_error), which nanobind surfaces as RuntimeError. Accept either so
    # this test is meaningful under both.
    GuardError = (LogicError, RuntimeError)

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
                            task = CopyTask(
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
                                task = CopyTask(
                                    MemAddress(dst_base + dst_buf_size * b),
                                    MemAddress(src_base + src_buf_size * b),
                                )
                                tasks.append(task)
                    batched_copy(CacheTier.GPU_MEM, CacheTier.GPU_MEM, num_bytes, tasks, stream)

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
        host_quota: int | None = None,
        num_attn_layers: int = 2,
        num_ssm_layers: int = 2,
        ssm_buffer_size: int = 8192,
        max_util_for_resume: float = 0.97,
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
                        BufferConfig(role=DataRole("ssm_state"), size=ssm_buffer_size),
                    ],
                )
            )
            lid += 1
        cache_tiers = [GpuCacheTierConfig(quota=gpu_quota)]
        if host_quota is not None:
            cache_tiers.append(HostCacheTierConfig(quota=host_quota))
        return KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=cache_tiers,
            layers=layers,
            max_util_for_resume=max_util_for_resume,
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

    @requires_cpp_backend
    def test_resume_protects_resident_reuse_source_before_allocating(self) -> None:
        """A deferred-state allocation must not evict its own reuse source."""
        slot_size = 2 << 20
        cfg = self._make_ssm_config(
            gpu_quota=2 * slot_size,
            host_quota=slot_size,
            num_attn_layers=0,
            num_ssm_layers=1,
            ssm_buffer_size=slot_size,
            max_util_for_resume=1.0,
        )
        self.manager = KVCacheManager(cfg)
        gpu_stats = _introspection.storage_statistics(self.manager, GPU_LEVEL)[0]
        host_level = CacheLevel(1)
        host_stats = _introspection.storage_statistics(self.manager, host_level)[0]
        self.assertEqual(gpu_stats.total, 2)
        self.assertEqual(host_stats.total, 1)

        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        prompt = [self.next_token() for _ in range(cfg.tokens_per_block)]

        seed = self.manager.create_kv_cache(custom_priority_callback=lambda _ordinal, _lc: 0)
        self.assertTrue(seed.resume(stream))
        seed.capacity = len(prompt)
        seed.history_length = len(prompt)
        seed.commit(prompt, is_end=True)
        seed.close()

        reused = self.manager.create_kv_cache(
            input_tokens=prompt,
            custom_priority_callback=lambda _ordinal, _lc: 0,
        )
        pressure = self.manager.create_kv_cache(custom_priority_callback=lambda _ordinal, _lc: 100)
        caches = [reused, pressure]
        try:
            self.assertEqual(reused.num_committed_tokens, len(prompt))
            self.assertTrue(pressure.resume(stream))
            pressure.stop_committing()
            pressure.suspend()

            gpu_before = _introspection.storage_statistics(self.manager, GPU_LEVEL)[0]
            host_before = _introspection.storage_statistics(self.manager, host_level)[0]
            reused_pages, _ = _introspection.active_page_stats(reused)
            self.assertEqual(gpu_before.free, 0)
            self.assertEqual(host_before.free, 1)
            self.assertEqual(reused_pages[GPU_LEVEL], 1)

            # The reuse source has lower eviction priority than the pressure page. Without the
            # preflight, newGpuSlots() moves that source to the only host slot and activate()
            # cannot bring it back. Protecting the source makes the pressure page the victim.
            self.assertTrue(reused.resume(stream))
        finally:
            for kv_cache in reversed(caches):
                if kv_cache.status != _KVCache.Status.CLOSED:
                    kv_cache.close()
            stream_holder.synchronize()

    @requires_cpp_backend
    def test_failed_offload_keeps_gpu_recurrent_pages_evictable(self) -> None:
        """A full host tier must not drain the GPU eviction queue."""
        host_level = CacheLevel(1)
        cfg = self._make_ssm_config(
            gpu_quota=8 << 20,
            host_quota=4 << 20,
            num_attn_layers=0,
            num_ssm_layers=1,
            ssm_buffer_size=1 << 20,
            max_util_for_resume=1.0,
        )
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        gpu_slots = _introspection.storage_statistics(self.manager, GPU_LEVEL)[0].total
        host_slots = _introspection.storage_statistics(self.manager, host_level)[0].total
        caches = []

        try:
            for _ in range(gpu_slots + host_slots):
                kv_cache = self.manager.create_kv_cache()
                caches.append(kv_cache)
                self.assertTrue(kv_cache.resume(stream))
                kv_cache.stop_committing()
                kv_cache.suspend()

            gpu_before = _introspection.storage_statistics(self.manager, GPU_LEVEL)[0]
            host_before = _introspection.storage_statistics(self.manager, host_level)[0]
            self.assertEqual(gpu_before.free, 0)
            self.assertEqual(host_before.free, 0)
            self.assertGreater(gpu_before.evictable, 0)

            blocked = self.manager.create_kv_cache()
            caches.append(blocked)
            self.assertFalse(blocked.resume(stream))
            gpu_after = _introspection.storage_statistics(self.manager, GPU_LEVEL)[0]
            host_after = _introspection.storage_statistics(self.manager, host_level)[0]

            self.assertEqual(gpu_after.evictable, gpu_before.evictable)
            self.assertEqual(host_after.free, host_before.free)
            self.assertEqual(host_after.evictable, host_before.evictable)
        finally:
            for kv_cache in reversed(caches):
                kv_cache.close()
            stream_holder.synchronize()

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
        iteration stats (TRTLLM-15217). Runs against the selected backend, so
        it checks the default C++ implementation and Python-backend parity.
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

    def test_num_tokens_before_hybrid_pruning_isolates_recurrent_truncation(self) -> None:
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
        self.assertEqual(kv._get_num_tokens_before_hybrid_pruning(), 48)
        kv.resume(stream)
        kv.close()

        # When the snapshot and the attention match agree, the diagnostic must
        # collapse onto num_committed_tokens rather than reporting the lookup.
        kv = self.manager.create_kv_cache(input_tokens=prompt[:64])
        self.assertEqual(kv.num_committed_tokens, 64)
        self.assertEqual(kv._get_num_tokens_before_hybrid_pruning(), 64)
        kv.resume(stream)
        kv.close()

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

        kv_cache = self.manager.create_kv_cache()
        kv_cache.resume(stream)
        kv_cache.capacity = len(prompt)
        kv_cache.history_length = len(prompt)

        attn_lc_id = _introspection.attention_life_cycle_ids(self.manager)[0]
        ssm_lc_id = _introspection.ssm_life_cycle_id(self.manager)
        assert ssm_lc_id is not None
        attn_tail_slot = kv_cache.get_base_page_indices(LayerGroupId(attn_lc_id))[1]
        ssm_slot = kv_cache.get_ssm_block_base_index(LayerGroupId(ssm_lc_id))

        kv_cache.commit(prompt, is_end=True)
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
        ssm_pg = _introspection.pool_group_index(manager, ssm_lc)
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

    def test_ratio_slot_count_rounding_matches_python(self):
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
            tuple(stat_slot_sizes(stat)): stat.total
            for stat in _introspection.storage_statistics(manager)
        }
        self.assertEqual(slots_by_size[(grain - 1,)], 2)
        self.assertEqual(slots_by_size[(grain,)], 3)
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
        stats = _introspection.storage_statistics(manager)
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
        stats = _introspection.storage_statistics(manager)
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
            self.assertIsNotNone(kv.get_scratch_desc(lg_id))
            self.assertTrue(kv.has_scratch_slots)
            with self.assertRaisesRegex(ValueError, "scratch blocks are needed"):
                kv.enable_swa_scratch_reuse = False

            self.engine.execute([Step(kv, prompt, history)], stream)
            kv.commit(prompt)
            history.extend(prompt)
            self.assertIsNone(kv.get_scratch_desc(lg_id))
            self.assertFalse(kv.has_scratch_slots)

            kv.enable_swa_scratch_reuse = False
            self.assertFalse(kv.enable_swa_scratch_reuse)
            kv.capacity = len(history) + 1
            self.assertFalse(kv.has_scratch_slots)
            self.assertIsNone(kv.get_scratch_desc(lg_id))

            self.engine.execute([Step(kv, [decode_token], history)], stream)
            kv.commit([decode_token])
            history.append(decode_token)
            self.assertFalse(kv.has_scratch_slots)

            kv.enable_swa_scratch_reuse = True
            self.assertTrue(kv.enable_swa_scratch_reuse)
            self.assertTrue(kv.resize(len(history) + len(second_prompt), len(history)))
            self.assertIsNotNone(kv.get_scratch_desc(lg_id))
            self.assertTrue(kv.has_scratch_slots)

            self.engine.execute([Step(kv, second_prompt, history)], stream)
            kv.commit(second_prompt)
            history.extend(second_prompt)
            self.assertIsNone(kv.get_scratch_desc(lg_id))
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

            scratch_desc = kv.get_scratch_desc(lg_id)
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

            # Scratch slot count in ScratchDesc matches expected
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
        """Verify that scratch blocks share coalesced slot IDs via ScratchDesc."""
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
            scratch_desc = kv.get_scratch_desc(lg_id)
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
            page_indices = converter(indices, PageIndexMode.PER_LAYER, scratch_desc)
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

            scratch_desc_0 = kv.get_scratch_desc(lg_id_0)
            self.assertIsNotNone(scratch_desc_0)
            self.assertEqual(scratch_desc_0.range.beg, chunk1_scratch_range[0])
            self.assertEqual(scratch_desc_0.range.end, chunk1_scratch_range[1])
            self.assertEqual(
                len(scratch_desc_0.slot_ids),
                chunk1_scratch_range[1] - chunk1_scratch_range[0],
            )

            # Layer 1 should have 0 scratch blocks
            scratch_desc_1 = kv.get_scratch_desc(lg_id_1)
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

            scratch_desc_0 = kv.get_scratch_desc(lg_id_0)
            if chunk2_scratch_range is None:
                self.assertIsNone(scratch_desc_0)
            else:
                self.assertIsNotNone(scratch_desc_0)
                self.assertEqual(scratch_desc_0.range.beg, chunk2_scratch_range[0])
                self.assertEqual(scratch_desc_0.range.end, chunk2_scratch_range[1])

            # Layer 1 should still have 0 scratch blocks
            scratch_desc_1 = kv.get_scratch_desc(lg_id_1)
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

    def prepare_partial(self, gpu_quota: int = 64 << 20, window_size: int | None = None) -> None:
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
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

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
        self.prepare_partial()
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

    TestSlotAllocatorShrink below pokes the Python SlotAllocator directly, so it
    never reaches the backend selected by TLLM_KV_CACHE_MANAGER_V2_BACKEND. This
    class goes through the manager instead, covering
    need_adjustment -> adjust() -> adjust_cache_level -> shrink/expand_pool_group
    on whichever backend is active (C++ by default).
    """

    _TOKENS_PER_BLOCK = 32
    _PROMPT_LEN = 64
    _DECODE_LEN = 96

    def prepare_two_pool_groups(self, gpu_quota: int = 256 << 20) -> None:
        """Two attention life cycles with different slot sizes -> two pool groups.

        Pool groups are formed per distinct slot layout, so differing window
        sizes alone are not enough -- the buffer sizes have to differ too.
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
        return [s.total for s in _introspection.storage_statistics(self.manager, GPU_LEVEL)]

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


class TestSlotAllocatorShrink(unittest.TestCase):
    def test_shrink_underused_pool(self) -> None:
        # Regression for NVBug 6225866: shrinking a pool whose new size is
        # still above the slot-ID high-water mark used to assert because
        # _num_active_slots - _target_capacity went negative.
        allocator = SlotAllocator(capacity=184064)
        slots = [allocator.allocate() for _ in range(2048)]
        for s in slots:
            allocator.release(s)
        self.assertEqual(allocator._num_active_slots, 2048)

        allocator.prepare_for_shrink(122624)
        self.assertEqual(len(allocator._overflow_slots), 0)
        self.assertTrue(allocator.finish_shrink())
        self.assertEqual(allocator._capacity, 122624)
        self.assertEqual(allocator._num_active_slots, 2048)
        self.assertFalse(allocator.shrink_in_progress)

    def test_shrink_touched_pool(self) -> None:
        # Sanity-check that the non-trivial migration path still works:
        # all ids are issued, half released, shrink to half.
        allocator = SlotAllocator(capacity=16)
        slots = [allocator.allocate() for _ in range(16)]
        for s in slots[8:]:
            allocator.release(s)
        self.assertEqual(allocator._num_active_slots, 16)

        allocator.prepare_for_shrink(8)
        self.assertEqual(len(allocator._overflow_slots), 8)
        self.assertTrue(allocator.finish_shrink())
        self.assertEqual(allocator._capacity, 8)
        self.assertEqual(allocator._num_active_slots, 8)

        for s in slots[:8]:
            allocator.release(s)


@pytest.mark.cpu_only
class TestBlockKeyHashing(unittest.TestCase):
    """Verify Hasher.update produces bit-identical digests to the per-token reference (no GPU needed)."""

    @staticmethod
    def _ref_update(seed: bytes, block: "list[int | bytes]") -> bytes:
        h = hashlib.sha256()
        h.update(seed)
        for item in block:
            # Normal token ids are packed as 4 little-endian bytes (31-bit range),
            # matching the C++ backend's 4-byte TokenIdExt layout.
            h.update(item.to_bytes(4, "little") if type(item) is int else item)
        return h.digest()

    def test_update_int_block_matches_reference(self) -> None:
        rng = random.Random(123)
        seed = b"\xaa\xbb\xcc"
        for n in (0, 1, 7, 32, 33, 257):
            block = [rng.randint(0, (1 << 31) - 1) for _ in range(n)]
            self.assertEqual(
                Hasher(seed).update(block).digest,
                self._ref_update(seed, block),
                f"int block of length {n}",
            )

    def test_update_mixed_multimodal_block(self) -> None:
        block = [randbytes(32), 5, 6, randbytes(32)] + list(range(20))
        seed = b"\x01"
        self.assertEqual(Hasher(seed).update(block).digest, self._ref_update(seed, block))

    def test_multimodal_digest_requires_sha256_length(self) -> None:
        for digest_size in (31, 33):
            with self.subTest(digest_size=digest_size), self.assertRaises(ValueError):
                gen_multimodal_cache_key_tokens(100, bytes(digest_size), 1)


if __name__ == "__main__":
    unittest.main()
