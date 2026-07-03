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
"""Executor-side tests for the contiguous primary KV cache prototype
(``KvCacheConfig.use_contiguous_kv_arena``; see
``contiguous_primary_kvcache/DESIGN.md``). Manager-internal behavior is
covered by ``tests/unittest/kv_cache_manager_v2_tests/test_contiguous_arena.py``;
this file covers the KVCacheManagerV2 adapter plumbing."""

import unittest
from unittest import mock

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import WriteThroughPolicy

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")

MiB = 1 << 20

NUM_LAYERS = 4
NUM_KV_HEADS = 4
HEAD_DIM = 128
TOKENS_PER_BLOCK = 8
MAX_SEQ_LEN = 256
MAX_TOKENS = 2048
# fp16 K+V: heads * dim * 2 roles * 2 bytes, per token per layer
BYTES_PER_TOKEN = NUM_KV_HEADS * HEAD_DIM * 2 * 2 * NUM_LAYERS
BLOCK_COLUMN_BYTES = BYTES_PER_TOKEN * TOKENS_PER_BLOCK  # all pool groups
GPU_QUOTA = MAX_TOKENS * BYTES_PER_TOKEN


def _create_arena_manager(**kv_cache_kwargs) -> KVCacheManagerV2:
    kv_cache_config = KvCacheConfig(
        max_tokens=MAX_TOKENS,
        enable_block_reuse=True,
        host_cache_size=64 * MiB,
        use_contiguous_kv_arena=True,
        **kv_cache_kwargs,
    )
    return KVCacheManagerV2(
        kv_cache_config,
        CacheType.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
    )


class TestKvCacheConfigArenaField(unittest.TestCase):
    def test_default_off(self) -> None:
        cfg = KvCacheConfig()
        self.assertFalse(cfg.use_contiguous_kv_arena)
        self.assertFalse(cfg.use_kv_cache_manager_v2)

    def test_arena_implies_v2(self) -> None:
        cfg = KvCacheConfig(use_contiguous_kv_arena=True)
        self.assertTrue(cfg.use_kv_cache_manager_v2)

    def test_env_knobs(self) -> None:
        env = {
            "TRTLLM_KV_ARENA_PHYS_PAGE_SIZE_MB": "16",
            "TRTLLM_KV_ARENA_MAP_AHEAD_PAGES": "3",
            "TRTLLM_KV_ARENA_WRITE_THROUGH": "on_commit",
        }
        with mock.patch.dict("os.environ", env):
            arena_cfg = KVCacheManagerV2._build_arena_config()
        self.assertEqual(arena_cfg.phys_page_size, 16 * MiB)
        self.assertEqual(arena_cfg.map_ahead_pages, 3)
        self.assertEqual(arena_cfg.write_through, WriteThroughPolicy.ON_COMMIT)

    def test_env_defaults(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            arena_cfg = KVCacheManagerV2._build_arena_config()
        self.assertEqual(arena_cfg.phys_page_size, 2 * MiB)
        self.assertEqual(arena_cfg.write_through, WriteThroughPolicy.ON_FREE)


@requires_cuda
class TestArenaAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = _create_arena_manager()

    def tearDown(self) -> None:
        self.mgr.shutdown()
        del self.mgr

    def test_arena_mode_enabled(self) -> None:
        self.assertTrue(self.mgr._arena_enabled)
        self.assertTrue(self.mgr.impl._storage.is_arena_mode)

    def test_capacity_accounting_is_physical(self) -> None:
        # get_num_free_blocks counts what the page budget can back, not VA
        budget = self.mgr.impl._storage.gpu_page_budget
        page_size = self.mgr.impl._storage.arena_config.phys_page_size
        expected_blocks = budget.total_pages * page_size // BLOCK_COLUMN_BYTES
        self.assertEqual(self.mgr.get_num_free_blocks(), expected_blocks)
        self.assertEqual(budget.total_pages, GPU_QUOTA // page_size)
        # clamp is page-based too and must not exceed physical capacity
        available = self.mgr.get_num_available_tokens(token_num_upper_bound=10**6)
        self.assertLessEqual(available, budget.total_pages * page_size // BYTES_PER_TOKEN)
        self.assertGreater(available, 0)

    def test_request_lifecycle_through_adapter(self) -> None:
        mgr = self.mgr
        budget = mgr.impl._storage.gpu_page_budget
        kv = mgr._create_kv_cache(1, None, None)
        self.assertIsNotNone(kv)
        # the VA reservation is sized for the per-request maximum
        self.assertEqual(kv._max_capacity, mgr.max_blocks_per_seq * TOKENS_PER_BLOCK)
        kv.cuda_stream = mgr._stream.cuda_stream
        self.assertTrue(kv.resume(mgr._stream.cuda_stream))
        self.assertTrue(kv.resize(3 * TOKENS_PER_BLOCK))
        # the adapter pre-sizes index buffers to max_blocks_per_seq; only the
        # first num_blocks entries are valid -- and they must be consecutive
        indices = list(kv.get_base_page_indices(0))[:3]
        self.assertEqual(indices, list(range(indices[0], indices[0] + 3)))
        self.assertGreater(budget.used_pages, 0)
        # suspend/resume via the adapter-facing kv methods
        kv.suspend()
        self.assertTrue(kv.resume(mgr._stream.cuda_stream))
        kv.close()
        mgr.kv_cache_map.pop(1)
        mgr.index_mapper.remove_sequence(1)
        torch.cuda.synchronize()
        self.assertGreaterEqual(mgr.impl.drain_gpu_reclaim(), 1)
        self.assertEqual(budget.used_pages, 0)


if __name__ == "__main__":
    unittest.main()
