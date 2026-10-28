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
"""KVCacheManagerV2 wrapper paths that build a ReuseScope for a request.

Every probe, prefetch and mirror creation must carry the request's locality
domain, or the native manager rejects the scope (localized) and the lookup
misses the domain-keyed blocks. The wrapper is driven as a bare instance over a
real native manager in mock localization mode, with no model.
"""

import gc
import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    CudaStream,
    DataRole,
    GpuCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    ReuseScope,
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


@contextmanager
def _mock_localization():
    env_backup = os.environ.get("TRT_LLM_MOCK_LOCALIZATION_SUPPORT")
    os.environ["TRT_LLM_MOCK_LOCALIZATION_SUPPORT"] = "1"
    try:
        yield
    finally:
        if env_backup is None:
            os.environ.pop("TRT_LLM_MOCK_LOCALIZATION_SUPPORT", None)
        else:
            os.environ["TRT_LLM_MOCK_LOCALIZATION_SUPPORT"] = env_backup


def _make_impl() -> KVCacheManager:
    config = KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=256 << 20, enable_locality_domains=True)],
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[
                    BufferConfig(role=DataRole("key"), size=8192),
                    BufferConfig(role=DataRole("value"), size=8192),
                ],
                sliding_window_size=None,
            )
        ],
    )
    with _mock_localization():
        return KVCacheManager(config)


class _RecordingImpl:
    """Delegates to the native manager and records every ReuseScope it sees."""

    def __init__(self, impl):
        self._impl = impl
        self.scopes = []

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def create_kv_cache(self, scope, *args, **kwargs):
        self.scopes.append(scope)
        return self._impl.create_kv_cache(scope, *args, **kwargs)

    def probe_reuse(self, scope, tokens):
        self.scopes.append(scope)
        return self._impl.probe_reuse(scope, tokens)

    def probe_first_new_block_key(self, scope, tokens):
        self.scopes.append(scope)
        return self._impl.probe_first_new_block_key(scope, tokens)

    def prefetch_reuse(self, scope, tokens, cache_level):
        # The fixture has no tier to prefetch from; only the scope matters here.
        self.scopes.append(scope)
        return False


def _request(request_id: int, domain: int):
    return SimpleNamespace(
        py_request_id=request_id, py_locality_domain_id=domain, lora_task_id=None, cache_salt=None
    )


def _bare_wrapper(impl) -> KVCacheManagerV2:
    wrapper = cast(KVCacheManagerV2, object.__new__(KVCacheManagerV2))
    wrapper.impl = impl
    wrapper.enable_block_reuse = True
    wrapper.can_evict = True
    wrapper.kv_cache_map = {}
    wrapper.tokens_per_block = impl.tokens_per_block
    wrapper._next_locality_domain = 0
    wrapper._locality_domain_load = [0, 0]
    return wrapper


class TestWrapperScopesCarryTheDomain(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.impl = _RecordingImpl(_make_impl())
        self.wrapper = _bare_wrapper(self.impl)
        self.tokens = list(range(64))
        self.wrapper._context_reuse_tokens = lambda req, reuse_limit=None: self.tokens[:-1]
        # Commit a prefix in domain 0 only.
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            native = self.impl._impl.create_kv_cache(
                ReuseScope(locality_domain_id=0), self.tokens[:-1]
            )
            self.assertTrue(native.resume(stream))
            self.assertTrue(native.resize(128))
            native.commit(self.tokens[native.num_committed_tokens :])
            native.stop_committing()
        native.close()
        self.impl.scopes.clear()

    def tearDown(self) -> None:
        gc.enable()
        self.impl._impl.shutdown()

    def test_probe_context_reuse_sees_only_its_domain(self) -> None:
        self.assertGreater(self.wrapper.probe_context_reuse(_request(1, 0)), 0)
        self.assertEqual(self.wrapper.probe_context_reuse(_request(2, 1)), 0)
        self.assertEqual([s.locality_domain_id for s in self.impl.scopes], [0, 1])

    def test_probe_prefix_match_length_takes_the_best_domain(self) -> None:
        match = self.wrapper.probe_prefix_match_length(self.tokens[:-1])
        self.assertGreater(match, 0)
        self.assertEqual(sorted(s.locality_domain_id for s in self.impl.scopes), [0, 1])

    def test_probe_first_new_block_key_uses_the_domain(self) -> None:
        self.wrapper.probe_first_new_block_key(_request(3, 1))
        self.assertEqual([s.locality_domain_id for s in self.impl.scopes], [1])

    def test_placement_sends_a_request_to_the_domain_holding_its_prefix(self) -> None:
        # Round-robin would start at domain 0, so place another request first to move it to 1.
        self.assertEqual(self.wrapper.pick_locality_domain(), 0)
        self.assertEqual(self.wrapper.pick_locality_domain(self.tokens), 0)
        self.assertEqual(self.wrapper.pick_locality_domain([10**6 + t for t in self.tokens]), 1)

    def test_prefetch_builds_a_domain_scope(self) -> None:
        self.wrapper.prefetch_for_context_tokens([_request(4, 1), _request(5, 0)])
        self.assertEqual([s.locality_domain_id for s in self.impl.scopes], [1, 0])


if __name__ == "__main__":
    unittest.main()
