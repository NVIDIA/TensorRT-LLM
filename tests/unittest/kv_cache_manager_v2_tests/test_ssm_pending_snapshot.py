# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pending SSM snapshot pages of ``_KVCache``.

``acquire_ssm_snapshot_slot()`` reserves a GPU SSM page that a forward step fills
with the recurrent state at a point inside the step; ``commit(...,
ssm_from_pending=True)`` ending at that point moves the page into the radix tree
instead of copying the live SSM page there. These tests write the pending page
the way such a step would (FakeEngine's SSM value of the last token before the
point) and check page ownership, reuse data, and every release path.
"""

import gc
import itertools
import os
import sys
import unittest
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        DEFAULT_BEAM_INDEX,
        AttentionLayerConfig,
        BufferConfig,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        LogicError,
        MemAddress,
        PageIndexMode,
        ReuseScope,
        SsmLayerConfig,
        TokenId,
        TokenIdExt,
        _introspection,
    )
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        DEFAULT_BEAM_INDEX,
        AttentionLayerConfig,
        BufferConfig,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        LogicError,
        MemAddress,
        PageIndexMode,
        ReuseScope,
        SsmLayerConfig,
        TokenId,
        TokenIdExt,
        _introspection,
    )

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# cuda_test_utils supplies temporary_sys_path, so its own path entry is added and
# removed by hand here; every later sibling import goes through that helper.
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from cuda_test_utils import (  # noqa: E402
        CachedCudaStream,
        exact_div,
        init_cuda_once,
        temporary_sys_path,
    )
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

with temporary_sys_path(_TEST_DIR):
    from fake_engine import FakeEngine, Step
    from kernels import fill_values


class TestPendingSsmSnapshot(unittest.TestCase):
    TOKENS_PER_BLOCK = 32

    def setUp(self) -> None:
        init_cuda_once()
        self._token_ids = itertools.count()
        gc.collect()
        gc.disable()
        self.stream_holder = CachedCudaStream()
        self.stream = cast(CudaStream, self.stream_holder.handle)

    def tearDown(self) -> None:
        gc.enable()
        if hasattr(self, "manager"):
            self.manager.shutdown()
            del self.manager

    def _config(
        self,
        gpu_quota: int = 32 << 20,
        enable_partial_reuse: bool = False,
        max_util_for_resume: float = 0.97,
    ) -> KVCacheManagerConfig:
        layers = []
        for layer_id in range(2):
            layers.append(
                AttentionLayerConfig(
                    layer_id=LayerId(layer_id),
                    buffers=[
                        BufferConfig(role=DataRole("key"), size=8192),
                        BufferConfig(role=DataRole("value"), size=8192),
                    ],
                )
            )
        for layer_id in range(2, 4):
            layers.append(
                SsmLayerConfig(
                    layer_id=LayerId(layer_id),
                    buffers=[BufferConfig(role=DataRole("ssm_state"), size=8192)],
                )
            )
        return KVCacheManagerConfig(
            tokens_per_block=self.TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota)],
            layers=layers,
            max_util_for_resume=max_util_for_resume,
            enable_partial_reuse=enable_partial_reuse,
            commit_min_snapshot=True,
        )

    def _setup(self, **config_kwargs) -> None:
        self.cfg = self._config(**config_kwargs)
        self.manager = KVCacheManager(self.cfg)
        self.engine = FakeEngine(self.cfg)
        self.ssm_lc = _introspection.ssm_life_cycle_id(self.manager)
        assert self.ssm_lc is not None
        self.ssm_pg = self.manager.get_life_cycle_pool_group_indices()[self.ssm_lc]

    def _tokens(self, n: int) -> list[TokenIdExt]:
        return [TokenId(next(self._token_ids)) for _ in range(n)]

    def _ssm_free_slots(self) -> int:
        return self.manager.get_storage_statistics()[self.ssm_pg].free

    def _ssm_layer_group(self) -> int:
        return self.manager.get_layer_group_id(LayerId(2))

    def _tree_ssm_page(self, tokens: list[TokenIdExt]) -> tuple[int, int | None] | None:
        """``(slot id, token count)`` of the SSM page of the last tree block matching ``tokens``."""
        num_tokens, pages = _introspection.reuse_match_pages(
            self.manager, ReuseScope(), tokens, self.ssm_lc, self.manager.enable_partial_match
        )
        self.assertEqual(num_tokens, len(tokens))
        return pages[-1]

    def _write_ssm_slot(self, kv_cache, slot: int, token: TokenIdExt) -> None:
        """Write a slot the way FakeEngine writes the live SSM page after ``token``."""
        for layer in self.cfg.layers:
            if not isinstance(layer, SsmLayerConfig):
                continue
            lc_id = self.manager.get_layer_group_id(layer.layer_id)
            for buf_id, buf in enumerate(layer.buffers):
                index_mode = (
                    PageIndexMode.SHARED
                    if kv_cache.supports_index_mode(PageIndexMode.SHARED, lc_id)
                    else PageIndexMode.PER_LAYER
                )
                pool = self.manager.get_mem_pool_base_address(layer.layer_id, buf.role, index_mode)
                stride = self.manager.get_page_stride(layer.layer_id, buf.role)
                converter = self.manager.get_page_index_converter(layer.layer_id, buf.role)
                page = list(
                    converter(
                        [slot],
                        index_mode,
                        kv_cache.get_swa_scratch_range(lc_id) or None,
                        kv_cache.get_scratch_slots(lc_id),
                    )
                )[0]
                fill_values(
                    MemAddress(pool + stride * page),
                    exact_div(buf.size, 1),
                    1,
                    1,
                    layer.layer_id,
                    buf_id,
                    DEFAULT_BEAM_INDEX,
                    [token],
                    self.stream,
                )

    def _prefill(self, kv_cache, prompt: list[TokenIdExt]) -> None:
        kv_cache.capacity = len(prompt)
        self.engine.execute([Step(kv_cache, prompt, [])], self.stream)

    def _snapshot_step(self, kv_cache, prompt: list[TokenIdExt], point: int) -> int:
        """One step over the whole prompt that also writes a pending snapshot at ``point``."""
        self._prefill(kv_cache, prompt)
        slot = kv_cache.acquire_ssm_snapshot_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(kv_cache.pending_ssm_snapshot_slot, slot)
        self.assertNotEqual(slot, kv_cache.get_ssm_block_base_index(self._ssm_layer_group()))
        self._write_ssm_slot(kv_cache, slot, prompt[point - 1])
        return slot

    def _check_reuse(self, prompt: list[TokenIdExt], expected_reuse: int) -> None:
        kv_cache = self.manager.create_kv_cache(input_tokens=prompt)
        self.assertEqual(kv_cache.num_committed_tokens, expected_reuse)
        self.assertTrue(kv_cache.resume(self.stream))
        kv_cache.capacity = expected_reuse
        # Checks the attention pages and that the SSM page holds the snapshot.
        self.engine.execute([Step(kv_cache, [], prompt[:expected_reuse])], self.stream)
        kv_cache.close()

    def test_commit_moves_pending_page_into_tree(self) -> None:
        self._setup()
        prompt = self._tokens(100)
        point = 64
        kv_cache = self.manager.create_kv_cache()
        self.assertTrue(kv_cache.resume(self.stream))
        free_before = self._ssm_free_slots()
        slot = self._snapshot_step(kv_cache, prompt, point)
        self.assertEqual(self._ssm_free_slots(), free_before - 1)

        kv_cache.history_length = point
        kv_cache.commit(prompt[:point], ssm_from_pending=True)
        self.assertIsNone(kv_cache.pending_ssm_snapshot_slot)
        # Moved, not copied: no further slot was taken and the tree owns the pending slot.
        self.assertEqual(self._ssm_free_slots(), free_before - 1)
        self.assertEqual(self._tree_ssm_page(prompt[:point]), (slot, self.TOKENS_PER_BLOCK))
        # The live page still holds the state after the whole prompt.
        self.engine.execute([Step(kv_cache, [], prompt)], self.stream)
        kv_cache.history_length = len(prompt)
        kv_cache.stop_committing()
        kv_cache.close()

        self._check_reuse(prompt, point)

    def test_partial_block_snapshot_from_pending(self) -> None:
        self._setup(enable_partial_reuse=True)
        prompt = self._tokens(70)
        point = 48
        kv_cache = self.manager.create_kv_cache()
        self.assertTrue(kv_cache.resume(self.stream))
        self._snapshot_step(kv_cache, prompt, point)
        kv_cache.history_length = point
        kv_cache.commit(prompt[:point], ssm_from_pending=True)
        self.assertIsNone(kv_cache.pending_ssm_snapshot_slot)
        kv_cache.history_length = len(prompt)
        kv_cache.stop_committing()
        kv_cache.close()

        self._check_reuse(prompt, point)

    def test_pending_page_released_when_block_already_has_snapshot(self) -> None:
        self._setup()
        prompt = self._tokens(100)
        point = 64
        first = self.manager.create_kv_cache()
        self.assertTrue(first.resume(self.stream))
        self._prefill(first, prompt[:point])
        first.history_length = point
        first.commit(prompt[:point])
        first_tree_page = self._tree_ssm_page(prompt[:point])
        self.assertIsNotNone(first_tree_page)

        second = self.manager.create_kv_cache()
        self.assertTrue(second.resume(self.stream))
        free_before = self._ssm_free_slots()
        self._snapshot_step(second, prompt, point)
        second.history_length = point
        second.commit(prompt[:point], ssm_from_pending=True)
        self.assertIsNone(second.pending_ssm_snapshot_slot)
        self.assertEqual(self._ssm_free_slots(), free_before)
        self.assertEqual(self._tree_ssm_page(prompt[:point]), first_tree_page)
        for kv_cache in (first, second):
            kv_cache.stop_committing()
            kv_cache.close()
        self._check_reuse(prompt, point)

    def test_commit_without_snapshot_point_releases_pending_page(self) -> None:
        self._setup()
        prompt = self._tokens(100)
        kv_cache = self.manager.create_kv_cache()
        self.assertTrue(kv_cache.resume(self.stream))
        free_before = self._ssm_free_slots()
        self._snapshot_step(kv_cache, prompt, 64)
        # Empty commit: no block, no snapshot.
        kv_cache.commit([], ssm_from_pending=True)
        self.assertIsNone(kv_cache.pending_ssm_snapshot_slot)
        self.assertEqual(self._ssm_free_slots(), free_before)
        kv_cache.close()

    def test_release_paths(self) -> None:
        self._setup()
        prompt = self._tokens(100)
        kv_cache = self.manager.create_kv_cache()
        self.assertTrue(kv_cache.resume(self.stream))
        free_before = self._ssm_free_slots()

        self._snapshot_step(kv_cache, prompt, 64)
        kv_cache.release_ssm_snapshot_slot()
        self.assertIsNone(kv_cache.pending_ssm_snapshot_slot)
        self.assertEqual(self._ssm_free_slots(), free_before)
        kv_cache.release_ssm_snapshot_slot()  # no-op without a pending page

        self.assertIsNotNone(kv_cache.acquire_ssm_snapshot_slot())
        kv_cache.suspend()
        self.assertIsNone(kv_cache.pending_ssm_snapshot_slot)
        self.assertTrue(kv_cache.resume(self.stream))
        self.assertEqual(self._ssm_free_slots(), free_before)

        self.assertIsNotNone(kv_cache.acquire_ssm_snapshot_slot())
        kv_cache.stop_committing()
        self.assertIsNone(kv_cache.pending_ssm_snapshot_slot)
        self.assertEqual(self._ssm_free_slots(), free_before)
        # Committing has stopped: nothing can be committed with a new page.
        self.assertIsNone(kv_cache.acquire_ssm_snapshot_slot())
        kv_cache.close()

        other = self.manager.create_kv_cache()
        self.assertTrue(other.resume(self.stream))
        free_before = self._ssm_free_slots()
        self.assertIsNotNone(other.acquire_ssm_snapshot_slot())
        other.close()
        self.assertEqual(self._ssm_free_slots(), free_before + 1)

    def test_invalid_use(self) -> None:
        self._setup()
        prompt = self._tokens(64)
        suspended = self.manager.create_kv_cache()
        with self.assertRaises(LogicError):
            suspended.acquire_ssm_snapshot_slot()
        # Closing needs a CUDA stream once SSM pages exist.
        self.assertTrue(suspended.resume(self.stream))
        suspended.close()

        kv_cache = self.manager.create_kv_cache()
        self.assertTrue(kv_cache.resume(self.stream))
        kv_cache.capacity = 64
        kv_cache.history_length = 32
        with self.assertRaises(LogicError):
            kv_cache.commit(prompt[:32], ssm_from_pending=True)
        self.assertIsNotNone(kv_cache.acquire_ssm_snapshot_slot())
        with self.assertRaises(LogicError):
            kv_cache.acquire_ssm_snapshot_slot()
        with self.assertRaises(LogicError):
            kv_cache.commit(prompt[:32], is_end=True, ssm_from_pending=True)
        self.assertIsNotNone(kv_cache.pending_ssm_snapshot_slot)
        kv_cache.close()

    def test_acquire_without_free_page_returns_none(self) -> None:
        # Resume never refuses for utilization, so the SSM pool can be filled to the last page.
        self._setup(max_util_for_resume=1.0)
        kv_cache = self.manager.create_kv_cache()
        self.assertTrue(kv_cache.resume(self.stream))
        # Each empty active cache holds exactly one live SSM page and nothing is committed, so
        # once they take every free SSM page there is nothing left to allocate or evict.
        fillers = []
        while self._ssm_free_slots() > 0:
            filler = self.manager.create_kv_cache()
            self.assertTrue(filler.resume(self.stream))
            fillers.append(filler)
        self.assertIsNone(kv_cache.acquire_ssm_snapshot_slot())
        self.assertIsNone(kv_cache.pending_ssm_snapshot_slot)
        fillers.pop().close()
        self.assertIsNotNone(kv_cache.acquire_ssm_snapshot_slot())
        self.assertEqual(self._ssm_free_slots(), 0)
        kv_cache.close()
        for filler in fillers:
            filler.close()


if __name__ == "__main__":
    unittest.main()
