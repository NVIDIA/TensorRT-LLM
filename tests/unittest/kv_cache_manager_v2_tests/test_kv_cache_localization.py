# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Locality domain behavior of KVCacheManager and its KV caches.

Localized tests opt in with GpuCacheTierConfig(enable_locality_domains=True) and set
TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1, so the manager reports two domains backed by
ordinary device memory. Physical placement is therefore not verified here; the
metadata, addressing, reuse isolation and per-domain capacity are.
"""

import gc
import os
import sys
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        BAD_PAGE_INDEX,
        AttentionLayerConfig,
        BufferConfig,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        ReuseScope,
        _introspection,
        _KVCache,
    )
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        BAD_PAGE_INDEX,
        AttentionLayerConfig,
        BufferConfig,
        CudaStream,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        ReuseScope,
        _introspection,
        _KVCache,
    )

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from cuda_test_utils import TemporaryCudaStream, init_cuda_once  # noqa: E402
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

ROLE_KEY = DataRole("key")
ROLE_VALUE = DataRole("value")

_256MB = 256 * (1 << 20)
_TOKENS_PER_BLOCK = 32
_KV_BUF_SIZE = 8192


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


def _make_config(
    num_layers: int = 2,
    gpu_quota: int = _256MB,
    tokens_per_block: int = _TOKENS_PER_BLOCK,
    kv_buf_size: int = _KV_BUF_SIZE,
    enable_locality_domains: bool = False,
) -> KVCacheManagerConfig:
    return KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        cache_tiers=[
            GpuCacheTierConfig(quota=gpu_quota, enable_locality_domains=enable_locality_domains)
        ],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(i),
                buffers=[
                    BufferConfig(role=ROLE_KEY, size=kv_buf_size),
                    BufferConfig(role=ROLE_VALUE, size=kv_buf_size),
                ],
                sliding_window_size=None,
            )
            for i in range(num_layers)
        ],
    )


def _make_localized_manager(**kwargs) -> KVCacheManager:
    kwargs.setdefault("enable_locality_domains", True)
    config = _make_config(**kwargs)
    with _override_localization_mode("1"):
        return KVCacheManager(config)


def _make_non_localized_manager(**kwargs) -> KVCacheManager:
    config = _make_config(**kwargs)
    with _override_localization_mode("0"):
        return KVCacheManager(config)


def _scope(locality_domain_id: int | None) -> ReuseScope:
    return ReuseScope(locality_domain_id=locality_domain_id)


def _valid_page_indices(kv_cache: _KVCache, layer_group_id: int = 0) -> list[int]:
    return [idx for idx in kv_cache.get_base_page_indices(layer_group_id) if idx != BAD_PAGE_INDEX]


class _ManagerTestCase(unittest.TestCase):
    manager: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.manager = None

    def tearDown(self) -> None:
        gc.enable()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def _commit_and_close(self, kv_cache: _KVCache, tokens: list[int], capacity: int) -> None:
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(capacity))
            uncommitted = tokens[kv_cache.num_committed_tokens :]
            if uncommitted:
                kv_cache.commit(uncommitted)
            kv_cache.stop_committing()
        kv_cache.close()


class TestLocalityDomainId(_ManagerTestCase):
    def test_stored_on_creation(self) -> None:
        self.manager = _make_localized_manager()
        for domain in (0, 1):
            kv_cache = self.manager.create_kv_cache(_scope(domain))
            self.assertEqual(kv_cache.locality_domain_id, domain)
            self.assertEqual(kv_cache.reuse_scope.locality_domain_id, domain)
            kv_cache.close()

    def test_none_when_not_localized(self) -> None:
        self.manager = _make_non_localized_manager()
        kv_cache = self.manager.create_kv_cache()
        self.assertIsNone(kv_cache.locality_domain_id)
        self.assertEqual(len(kv_cache.reuse_scope), 2)
        kv_cache.close()

    def test_out_of_range_rejected(self) -> None:
        self.manager = _make_localized_manager()
        with self.assertRaises(Exception):
            self.manager.create_kv_cache(_scope(2))
        self.manager = _make_non_localized_manager()
        with self.assertRaises(Exception):
            self.manager.create_kv_cache(_scope(0))


class TestLocalizedQueries(_ManagerTestCase):
    def test_num_locality_domains(self) -> None:
        self.manager = _make_non_localized_manager()
        self.assertEqual(self.manager.num_locality_domains, 1)
        self.manager.shutdown()
        self.manager = _make_localized_manager()
        self.assertEqual(self.manager.num_locality_domains, 2)

    def test_free_slots_follow_the_domain(self) -> None:
        self.manager = _make_localized_manager()
        before = _introspection.locality_domain_free_slots(self.manager)
        self.assertEqual(len(before), 2)
        self.assertGreater(before[0], 0)
        self.assertEqual(before[0], before[1])

        kv_cache = self.manager.create_kv_cache(_scope(1))
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(64))
        during = _introspection.locality_domain_free_slots(self.manager)
        self.assertEqual(during[0], before[0])
        self.assertLess(during[1], before[1])

        kv_cache.close()
        self.assertEqual(_introspection.locality_domain_free_slots(self.manager), before)

    def test_free_slots_non_localized(self) -> None:
        self.manager = _make_non_localized_manager()
        free_slots = _introspection.locality_domain_free_slots(self.manager)
        self.assertEqual(len(free_slots), 1)
        self.assertGreater(free_slots[0], 0)

    def test_page_index_upper_bound_covers_both_domains(self) -> None:
        self.manager = _make_localized_manager()
        caches = [self.manager.create_kv_cache(_scope(domain)) for domain in (0, 1)]
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            for kv_cache in caches:
                self.assertTrue(kv_cache.resume(stream))
                self.assertTrue(kv_cache.resize(128))
        upper_bound = self.manager.get_page_index_upper_bound(LayerId(0), ROLE_KEY)
        indices = [idx for kv_cache in caches for idx in _valid_page_indices(kv_cache)]
        self.assertTrue(indices)
        self.assertLess(max(indices), upper_bound)
        self.assertEqual(len(set(indices)), len(indices))
        for kv_cache in caches:
            kv_cache.close()

    def test_mapped_buffer_views_cover_the_pool(self) -> None:
        self.manager = _make_localized_manager()
        views = self.manager.get_mapped_buffer_views(LayerId(0), ROLE_KEY)
        self.assertTrue(views)
        for address, num_pages in views:
            self.assertGreater(address, 0)
            self.assertGreater(num_pages, 0)
        self.manager.shutdown()
        self.manager = _make_non_localized_manager()
        self.assertEqual(len(self.manager.get_mapped_buffer_views(LayerId(0), ROLE_KEY)), 1)


class TestLocalizedReuse(_ManagerTestCase):
    def test_same_domain_reuses_blocks(self) -> None:
        self.manager = _make_localized_manager()
        tokens = list(range(64))
        kv_a = self.manager.create_kv_cache(_scope(0), tokens[:-1])
        self._commit_and_close(kv_a, tokens, 128)
        kv_b = self.manager.create_kv_cache(_scope(0), tokens[:-1])
        self.assertGreater(kv_b.num_committed_tokens, 0)
        self._commit_and_close(kv_b, tokens, 128)

    def test_other_domain_does_not_reuse_blocks(self) -> None:
        self.manager = _make_localized_manager()
        tokens = list(range(64))
        kv_a = self.manager.create_kv_cache(_scope(0), tokens[:-1])
        self._commit_and_close(kv_a, tokens, 128)
        kv_b = self.manager.create_kv_cache(_scope(1), tokens[:-1])
        self.assertEqual(kv_b.num_committed_tokens, 0)
        self._commit_and_close(kv_b, tokens, 128)
        kv_c = self.manager.create_kv_cache(_scope(1), tokens[:-1])
        self.assertGreater(kv_c.num_committed_tokens, 0)
        kv_c.close()

    def test_non_localized_reuse_unaffected(self) -> None:
        self.manager = _make_non_localized_manager()
        tokens = list(range(64))
        kv_a = self.manager.create_kv_cache(None, tokens[:-1])
        self._commit_and_close(kv_a, tokens, 128)
        kv_b = self.manager.create_kv_cache(None, tokens[:-1])
        self.assertGreater(kv_b.num_committed_tokens, 0)
        self._commit_and_close(kv_b, tokens, 128)


class TestLocalizedSuspendResume(_ManagerTestCase):
    def test_suspend_resume_preserves_domain_and_pages(self) -> None:
        self.manager = _make_localized_manager()
        kv_cache = self.manager.create_kv_cache(_scope(1))
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(128))
            before = _valid_page_indices(kv_cache)
            kv_cache.suspend()
            self.assertTrue(kv_cache.resume(stream))
        self.assertEqual(kv_cache.locality_domain_id, 1)
        self.assertEqual(_valid_page_indices(kv_cache), before)
        kv_cache.close()


if __name__ == "__main__":
    unittest.main()
