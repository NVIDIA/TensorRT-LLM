# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-tree test for KVCacheManagerV2.prefetch_to_gpu.

A waiting request's reusable prefix that was evicted to the host tier is found through a
transient, never-resumed cache and migrated back to the GPU. The tier of a held cache's pages
is only visible through ``cached_tokens_by_level`` (taken at match time): the page storage
snapshot reports no levels for a cache that is not active, which is why it cannot be used.
"""

import gc
import os
import sys
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 as kvm

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import GPU_LEVEL, CudaStream, KVCacheManager, ReuseScope, TokenId
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        GPU_LEVEL,
        CudaStream,
        KVCacheManager,
        ReuseScope,
        TokenId,
    )

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from cuda_test_utils import (  # noqa: E402
        TemporaryCudaStream,
        init_cuda_once,
        temporary_sys_path,
    )
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

with temporary_sys_path(_TEST_DIR):
    from test_kv_cache_manager_v2 import create_config

TOKENS_PER_BLOCK = 32
PROMPT_BLOCKS = 16
# 2 layers x (K + V) x 8 KiB per block: 32 KiB per block, so the GPU tier holds about 128 blocks.
GPU_QUOTA = 4 << 20
HOST_QUOTA = 64 << 20


class TestKvGpuPrefetchRuntime(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = KVCacheManager(
            create_config(
                tokens_per_block=TOKENS_PER_BLOCK,
                gpu_quota=GPU_QUOTA,
                host_quota=HOST_QUOTA,
                disk_quota=0,
                num_layers=2,
                window_size=None,
                sink_tokens=0,
            )
        )

    def tearDown(self) -> None:
        gc.enable()
        if hasattr(self, "manager"):
            self.manager.shutdown()
            del self.manager

    @staticmethod
    def tokens(start, count):
        return [TokenId(start + i) for i in range(count)]

    def commit(self, tokens):
        with TemporaryCudaStream([]) as stream_holder:
            stream = cast(CudaStream, stream_holder.handle)
            kv_cache = self.manager.create_kv_cache(ReuseScope(), tokens)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(len(tokens)))
            uncommitted = tokens[kv_cache.num_committed_tokens :]
            if uncommitted:
                kv_cache.commit(uncommitted)
            kv_cache.close()
        stream_holder.take_finish_event().synchronize()

    def levels(self, tokens):
        kv_cache = self.manager.create_kv_cache(ReuseScope(), tokens)
        try:
            return list(kv_cache.cached_tokens_by_level)
        finally:
            kv_cache.close()

    def kvm_view(self, tokens):
        return SimpleNamespace(
            enable_block_reuse=True,
            impl=self.manager,
            _context_reuse_tokens=lambda req: tokens,
            _derive_reuse_salt=lambda salt: None,
            _kv_token_bytes_estimate=lambda: 100,
        )

    def evict_to_host(self, prompt):
        self.commit(prompt)
        # Enough newer prompts to push the first one out of the GPU tier.
        n = PROMPT_BLOCKS * TOKENS_PER_BLOCK
        for i in range(1, 12):
            self.commit(self.tokens(i * 100_000, n))
        before = self.levels(prompt)
        self.assertGreater(sum(before[GPU_LEVEL + 1 :]), 0, before)
        return before

    def test_held_cache_snapshot_has_no_levels(self):
        prompt = self.tokens(0, PROMPT_BLOCKS * TOKENS_PER_BLOCK)
        self.evict_to_host(prompt)
        kv_cache = self.manager.create_kv_cache(ReuseScope(), prompt)
        try:
            levels = kv_cache.get_page_storage_snapshot(0).cache_levels
            self.assertTrue(all(level is None for level in levels), levels)
        finally:
            kv_cache.close()

    def test_prefetch_to_gpu_moves_host_prefix(self):
        prompt = self.tokens(0, PROMPT_BLOCKS * TOKENS_PER_BLOCK)
        before = self.evict_to_host(prompt)
        req = SimpleNamespace(cache_salt=None, lora_task_id=None)
        nbytes = kvm.KVCacheManagerV2.prefetch_to_gpu(self.kvm_view(prompt), req)
        self.assertEqual(nbytes, sum(before[GPU_LEVEL + 1 :]) * 100)
        after = self.levels(prompt)
        self.assertEqual(sum(after), sum(before))
        self.assertEqual(sum(after[GPU_LEVEL + 1 :]), 0, after)
        # Already on the GPU: nothing to move, no hold.
        self.assertEqual(kvm.KVCacheManagerV2.prefetch_to_gpu(self.kvm_view(prompt), req), 0)


if __name__ == "__main__":
    unittest.main()
