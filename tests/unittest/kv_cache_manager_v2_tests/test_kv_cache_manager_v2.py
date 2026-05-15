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
import itertools
import os
import random
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from random import randbytes
from statistics import median
from typing import TYPE_CHECKING, Iterator, NamedTuple, cast

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
        SsmLayerConfig,
        TokenId,
        TokenIdExt,
        _KVCache,
    )
    from kv_cache_manager_v2._block_radix_tree import traverse_post_order
    from kv_cache_manager_v2._common import (
        BAD_PAGE_INDEX,
        GPU_LEVEL,
        CacheTier,
        MemAddress,
        PageIndexMode,
        PageStatus,
        SlidingWindowSize,
    )
    from kv_cache_manager_v2._copy_engine import CopyTask, batched_copy
    from kv_cache_manager_v2._exceptions import OutOfPagesError
    from kv_cache_manager_v2._life_cycle_registry import SsmLifeCycle
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
        unwrap_rawref,
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
        SsmLayerConfig,
        TokenId,
        TokenIdExt,
        _KVCache,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import traverse_post_order
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
        BAD_PAGE_INDEX,
        GPU_LEVEL,
        CacheTier,
        MemAddress,
        PageIndexMode,
        PageStatus,
        SlidingWindowSize,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import CopyTask, batched_copy
    from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import SsmLifeCycle
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
        unwrap_rawref,
    )

from copy import deepcopy

from parameterized import parameterized

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from fake_engine import FakeEngine, Role, Step
    from kernels import enable_kernel_delay

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
        vocab_size=4096,
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
    ):
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
        return self.Request(
            req_id, self.manager.create_kv_cache(lora_task_id, prompt), prompt, decode_len
        )

    def run_request(self, req: Request, interval: int, refcheck: bool) -> float:
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

    def run_naive(
        self,
        seq_len: int,
        interval: int = 1,
        refcheck: bool = True,
        use_external_page_index_buf: bool = False,
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
            time_taken = self.run_request(req0, interval, refcheck)

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

        for root_block in self.manager._radix_tree.next.values():
            for block0 in root_block.next.values():
                for block in traverse_post_order(block0):
                    for page in block.storage:
                        if page is not None:
                            assert unwrap_rawref(page).status == PageStatus.DROPPABLE

        req0 = reusable_requests[0]
        prompt1 = req0.kv_cache._committed_tokens[: (seq_len // 2 - 7)]
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

    @parameterized.expand(list(itertools.product([False, True], repeat=2)))
    # @assert_no_ref_cycle
    def test_naive(self, use_external_page_index_buf: bool, use_block_quant: bool) -> None:
        self.prepare(
            256 << 20,
            256 << 20,
            1 << 30,
            36,
            128,
            48,
            block_quant_buf_size=(1024 if use_block_quant else None),
        )
        self.run_naive(512, 1, True, use_external_page_index_buf)

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
        kv_cache = self.manager.create_kv_cache(
            lora_task_id, prompt[:-1] if self.enable_reuse else None, id=next(self.req_id_gen)
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
                self.past_sequences.append(kv_cache._committed_tokens[:seq_len])
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
                if (
                    ok
                    and not self.enable_reuse
                    and kv_cache._commit_state == _KVCache.CommitState.ALLOWED
                ):
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
        kv_cache = self.manager.create_kv_cache(lora_task_id, prompt)
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
            vocab_size=1024,
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
            vocab_size=129280,
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
        # also shrink the host quota, this would evict some pages to disk
        success = self.manager.resize(HOST_LEVEL, 4 << 20)
        assert success and self.manager.get_quota(HOST_LEVEL) <= 4 << 20
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
            vocab_size=1024,
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
        ssm_reuse_interval: int = 512,
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
            vocab_size=1024,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=layers,
            ssm_reuse_interval=ssm_reuse_interval,
            enable_partial_reuse=False,
        )

    def test_suspend_and_resume_with_ssm(self) -> None:
        """Suspend and resume work correctly (SSM page locks/unlocks)."""
        cfg = self._make_ssm_config()
        self.manager = KVCacheManager(cfg)
        kv_cache = self.manager.create_kv_cache()
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache.resume(stream)
        ssm_lg = None
        for lc_id, lc in self.manager._life_cycles.items():
            if isinstance(lc, SsmLifeCycle):
                ssm_lg = LayerGroupId(lc_id)
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
        """input_tokens are accepted but no prefix reuse happens before first snapshot boundary."""
        cfg = self._make_ssm_config(tokens_per_block=32, ssm_reuse_interval=512)
        self.manager = KVCacheManager(cfg)
        # 64 tokens < ssm_reuse_interval=512, so no snapshot boundary reached → no SSM reuse
        tokens = [self.next_token() for _ in range(64)]
        kv_cache = self.manager.create_kv_cache(input_tokens=tokens)
        self.assertEqual(
            kv_cache.num_committed_tokens, 0, "No reuse before first snapshot boundary"
        )
        # Resume before close so cuda_stream is set
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)
        kv_cache.resume(stream)
        kv_cache.close()

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
        ssm_reuse_interval: int = 64,
        gpu_quota: int = 32 << 20,
        num_attn_layers: int = 2,
        num_ssm_layers: int = 2,
    ) -> KVCacheManagerConfig:
        return self._make_ssm_config(
            tokens_per_block=tokens_per_block,
            gpu_quota=gpu_quota,
            num_attn_layers=num_attn_layers,
            num_ssm_layers=num_ssm_layers,
            ssm_reuse_interval=ssm_reuse_interval,
        )

    def test_ssm_reuse_interval_boundary(self) -> None:
        """Snapshots only happen at interval boundaries, not every block."""
        tokens_per_block = 32
        ssm_reuse_interval = 128  # snapshot every 4 blocks
        cfg = self._make_ssm_reuse_config(
            tokens_per_block=tokens_per_block,
            ssm_reuse_interval=ssm_reuse_interval,
        )
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        # Commit 96 tokens (3 blocks) — no snapshot at interval 128
        prompt = [self.next_token() for _ in range(96)]
        kv1 = self.manager.create_kv_cache()
        kv1.resume(stream)
        kv1.capacity = len(prompt)
        kv1.history_length = len(prompt)
        kv1.commit(prompt)
        kv1.stop_committing()
        kv1.close()

        # Try to reuse — should get 0 since no snapshot exists
        kv2 = self.manager.create_kv_cache(input_tokens=prompt)
        self.assertEqual(
            kv2.num_committed_tokens, 0, "No reuse when no SSM snapshot at interval boundary"
        )
        kv2.resume(stream)
        kv2.close()

    def test_ssm_reuse_data_integrity(self) -> None:
        """After reuse, SSM data matches the snapshot (verified by FakeEngine)."""
        tokens_per_block = 32
        ssm_reuse_interval = 64
        cfg = self._make_ssm_reuse_config(
            tokens_per_block=tokens_per_block,
            ssm_reuse_interval=ssm_reuse_interval,
        )
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

    def test_ssm_reuse_config_validation(self) -> None:
        """Invalid ssm_reuse_interval raises assertion."""
        # Not divisible by tokens_per_block
        with self.assertRaises(AssertionError):
            self._make_ssm_config(tokens_per_block=32, ssm_reuse_interval=50)
        # Zero interval
        with self.assertRaises(AssertionError):
            self._make_ssm_config(tokens_per_block=32, ssm_reuse_interval=0)


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

    def _make_config(
        self,
        gpu_quota: int = 128 << 20,
        typical_step: BatchDesc | None = None,
        constraints: list[BatchDesc] | None = None,
        host_quota: int = 0,
        num_windowed_layers: int = 1,
        num_full_layers: int = 1,
        enable_swa_scratch_reuse: bool = False,
    ) -> KVCacheManagerConfig:
        """Create a config with two pool groups (windowed vs non-windowed).

        Uses large, non-power-of-2 buffer sizes so 2MB granularity rounding
        is non-trivial and constraint clamping is exercised.

        With num_windowed_layers / num_full_layers > 1 and
        enable_swa_scratch_reuse=True, multiple layers per lifecycle give
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
            vocab_size=4096,
            cache_tiers=cache_tiers,
            layers=layers,
            typical_step=typical_step,
            constraints=constraints or [],
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
        )

    def test_default_init_ratio(self):
        """Without typical_step or constraints, uses hardcoded fallback."""
        cfg = self._make_config()
        manager = KVCacheManager(cfg)
        ratio = manager._current_gpu_ratio
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
        ratio = manager._current_gpu_ratio
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
        ratio = manager._current_gpu_ratio
        self.assertEqual(len(ratio), 2)
        self.assertAlmostEqual(sum(ratio), 1.0, places=6)
        # Windowed layers (window=128) have many stale blocks, non-windowed keep all.
        self.assertLess(ratio[0], ratio[1])
        self.assertLess(ratio[0], 0.15)
        manager.shutdown()

    def test_constraints_floor_typical_step(self):
        """Constraints clamp the typical_step ratio from below."""
        typical = BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=4000)] * 32)
        constraint = BatchDesc(kv_caches=[KVCacheDesc(capacity=256, history_length=128)] * 256)
        cfg_unconstrained = self._make_config(typical_step=typical)
        mgr_unconstrained = KVCacheManager(cfg_unconstrained)
        ratio_unconstrained = mgr_unconstrained._current_gpu_ratio

        cfg_constrained = self._make_config(typical_step=typical, constraints=[constraint])
        mgr_constrained = KVCacheManager(cfg_constrained)
        ratio_constrained = mgr_constrained._current_gpu_ratio

        self.assertGreater(ratio_constrained[0], ratio_unconstrained[0])
        self.assertAlmostEqual(sum(ratio_constrained), 1.0, places=6)
        mgr_unconstrained.shutdown()
        mgr_constrained.shutdown()

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
        stats = manager._storage.get_statistics()
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
        ratio_no_constraint = mgr_no_constraint._current_gpu_ratio

        # Ratio with constraint that typical already covers.
        cfg_with_constraint = self._make_config(
            gpu_quota=gpu_quota,
            typical_step=typical,
            constraints=[constraint],
        )
        mgr_with_constraint = KVCacheManager(cfg_with_constraint)
        ratio_with_constraint = mgr_with_constraint._current_gpu_ratio

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
        Typical step: 4 prefill requests (history=0, capacity=4096).

        Without scratch: both PGs need the same block count; ratio reflects
        the buffer-size difference only.
        With scratch: PG0 needs far fewer slots -> ratio shifts toward PG1.
        """
        step = BatchDesc(kv_caches=[KVCacheDesc(capacity=4096, history_length=0)] * 4)
        multi = dict(num_windowed_layers=16, num_full_layers=16)
        cfg_no = self._make_config(typical_step=step, enable_swa_scratch_reuse=False, **multi)
        cfg_yes = self._make_config(typical_step=step, enable_swa_scratch_reuse=True, **multi)
        mgr_no = KVCacheManager(cfg_no)
        mgr_yes = KVCacheManager(cfg_yes)
        ratio_no = mgr_no._current_gpu_ratio
        ratio_yes = mgr_yes._current_gpu_ratio

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
        stats = manager._storage.get_statistics()
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
    ):
        """Prepare a manager with scratch reuse enabled."""
        kv_buf_size = 8192
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            vocab_size=4096,
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
            enable_swa_scratch_reuse=True,
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

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

    def test_scratch_shared_slot_ids(self):
        """Verify that scratch blocks share coalesced slot IDs via ScratchDesc."""
        # 8 layers, window=32, tokens_per_block=32, prompt=256
        # num_sub_pages = 8 (all layers in one group)
        # blocks 0-6 are scratch (7 blocks), block 7 is in-window (normal)
        # 7 scratch blocks / 8 sub_pages = 1 scratch slot
        self._prepare_scratch(num_layers=8, window_size=32, tokens_per_block=32, gpu_quota=16 << 20)

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
            self.assertEqual(num_scratch_blocks, 7)  # blocks 0-6

            # 7 blocks / 8 sub_pages = ceil = 1 scratch slot
            self.assertEqual(len(scratch_desc.slot_ids), 1)

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

    def test_scratch_chunk_size_variation(self):
        """Verify scratch block allocation with changing chunk sizes and multiple window sizes.

        This ensures both positive and negative net_alloc_counts code paths are tested
        simultaneously across different layers.

        Layer 0: window_size = 64 (2 blocks)
        Layer 1: window_size = 256 (8 blocks)

        Chunk 1: resize(256) -> 8 blocks.
          - Layer 0 (stale 0-6): needs 6 scratch blocks (net_alloc_counts = 6 > 0)
          - Layer 1 (stale 0-0): needs 0 scratch blocks (net_alloc_counts = 8 > 0)

        Chunk 2: resize(352, 256) -> 11 blocks.
          - Layer 0 (stale 0-9): needs 1 scratch block [8, 9). delta_scratch = -5. New normal = 2.
            net_alloc_counts = -3 < 0
          - Layer 1 (stale 0-3): needs 0 scratch blocks. delta_scratch = 0. New normal = 3.
            net_alloc_counts = 3 > 0
        """
        tokens_per_block = 32
        gpu_quota = 32 << 20
        kv_buf_size = 8192

        self.cfg = KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            vocab_size=4096,
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
            enable_swa_scratch_reuse=True,
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

            # Layer 0 should have 6 scratch blocks: range [0, 6)
            scratch_desc_0 = kv.get_scratch_desc(lg_id_0)
            self.assertIsNotNone(scratch_desc_0)
            self.assertEqual(scratch_desc_0.range.beg, 0)
            self.assertEqual(scratch_desc_0.range.end, 6)
            self.assertEqual(len(scratch_desc_0.slot_ids), 6)

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

            # Chunk 2: resize to 352 with history_length=256
            success = kv.resize(352, 256)
            self.assertTrue(success)

            # Layer 0 should have 1 scratch block: range [8, 9)
            scratch_desc_0 = kv.get_scratch_desc(lg_id_0)
            self.assertIsNotNone(scratch_desc_0)
            self.assertEqual(scratch_desc_0.range.beg, 8)
            self.assertEqual(scratch_desc_0.range.end, 9)

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


if __name__ == "__main__":
    unittest.main()
