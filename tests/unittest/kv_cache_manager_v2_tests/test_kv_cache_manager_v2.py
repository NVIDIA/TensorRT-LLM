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
        BufferConfig,
        BufferId,
        BufferSlice,
        CacheLevel,
        CudaStream,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerGroupId,
        LayerId,
        TokenId,
        TokenIdExt,
        _KVCache,
    )
    from kv_cache_manager_v2._block_radix_tree import traverse_post_order
    from kv_cache_manager_v2._common import (
        GPU_LEVEL,
        CacheTier,
        MemAddress,
        PageStatus,
        SlidingWindowSize,
    )
    from kv_cache_manager_v2._copy_engine import CopyTask, batched_copy
    from kv_cache_manager_v2._exceptions import OutOfPagesError
    from kv_cache_manager_v2._utils import (
        CachedCudaStream,
        TemporaryCudaStream,
        div_up,
        exact_div,
        init_cuda_once,
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
        BufferConfig,
        BufferId,
        BufferSlice,
        CacheLevel,
        CudaStream,
        DiskCacheTierConfig,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerGroupId,
        LayerId,
        TokenId,
        TokenIdExt,
        _KVCache,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import traverse_post_order
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
        GPU_LEVEL,
        CacheTier,
        MemAddress,
        PageStatus,
        SlidingWindowSize,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import CopyTask, batched_copy
    from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        CachedCudaStream,
        TemporaryCudaStream,
        div_up,
        exact_div,
        init_cuda_once,
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
                buffers=layer_buffers,
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
            page_indices = [
                array.array("i", [-1]) * max_num_blocks for _ in range(num_layer_groups)
            ]
            for id in range(num_layer_groups):
                req0.kv_cache.set_page_index_buf(
                    DEFAULT_BEAM_INDEX, LayerGroupId(id), memoryview(page_indices[id])
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
            page_indices = [
                array.array("i", [-1]) * max_num_blocks for _ in range(num_layer_groups)
            ]
            for id in range(num_layer_groups):
                req0.kv_cache.set_page_index_buf(
                    DEFAULT_BEAM_INDEX, LayerGroupId(id), memoryview(page_indices[id])
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

        def __init__(self, full_config: KVCacheManagerConfig, tp_size: int, pp_size: int):
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
                layers = full_layers[layer_start : layer_start + num_local_layers]
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
                    engine = FakeEngine(config)
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
        gpu_quota: int = 128 << 20,
        host_quota: int = 128 << 20,
        disk_quota: int = 0,
        num_layers: int = 4,
        window_size: SlidingWindowSize = 128,
        sink_tokens: int = 0,
        tokens_per_block: int = 32,
        kv_buf_size: int = 8192,
        block_quant_buf_size: int | None = None,
        prefill_pp_size: int = 1,
        prefill_tp_size: int = 1,
        decode_pp_size: int = 1,
        decode_tp_size: int = 1,
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
        self.prefill = self.NodeGroup(self.full_config, prefill_tp_size, prefill_pp_size)
        self.decode = self.NodeGroup(self.full_config, decode_tp_size, decode_pp_size)

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
                src_buffer_slices = (
                    BufferSlice(b, src_tp_slice.num_slices, src_tp_slice.slice_rank)
                    for b in buffers
                )
                dst_buffer_slices = (
                    BufferSlice(b, dst_tp_slice.num_slices, dst_tp_slice.slice_rank)
                    for b in buffers
                )
                src_pages = src.manager.get_aggregated_pages(src_buffer_slices)
                dst_pages = dst.manager.get_aggregated_pages(dst_buffer_slices)
                for src_page, dst_page in zip(src_pages, dst_pages, strict=True):
                    assert src_page.size == dst_page.size
                    num_bytes = src_page.size
                    assert all(
                        s.buffer_id == d.buffer_id
                        for s, d in zip(src_page.buffers, dst_page.buffers, strict=True)
                    )
                    tasks = [
                        CopyTask(
                            MemAddress(dst_page.base + dst_page.stride * i),
                            MemAddress(src_page.base + src_page.stride * j),
                        )
                        for i, j in zip(
                            dst.kv_cache.get_aggregated_page_indices(
                                dst_page.layer_group_id, valid_only=True
                            ),
                            src.kv_cache.get_aggregated_page_indices(
                                src_page.layer_group_id, valid_only=True
                            ),
                            strict=True,
                        )
                    ]
                    batched_copy(CacheTier.GPU_MEM, CacheTier.GPU_MEM, num_bytes, tasks, stream)

    @parameterized.expand([(1, 1, 1, 1), (1, 2, 1, 1), (1, 1, 1, 2), (2, 1, 1, 1), (1, 1, 2, 1)])
    def test_disaggregated_serving(
        self,
        prefill_tp_size: int,
        prefill_pp_size: int,
        decode_tp_size: int,
        decode_pp_size: int,
    ) -> None:
        self.prepare(prefill_tp_size, prefill_pp_size, decode_tp_size, decode_pp_size)

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


if __name__ == "__main__":
    unittest.main()
