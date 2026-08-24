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
"""Real-tree tests for the scheduler's first-new-block probe.

``_first_new_block_key`` (``tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py``)
names the block a request will commit next, so the scheduler can defer a second
request that would recompute the same prefix. These tests check that claim
against a live radix tree rather than against the formula itself:

* the key the probe names is the key of the block the commit actually stores
  (read back out of the KV cache event stream), and
* two requests sharing an uncached prefix probe to the same key, which stops
  being true once one of them commits.

Both run on a uniform full-attention layout and on a variable-window (VSWA)
layout, since v2 -- unlike v1, whose ``analyzePrefixReuse`` asserts on
variable-window managers -- supports VSWA here.
"""

import gc
import os
import unittest
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.runtime.kv_cache_hash import truncate_sha256_hash_to_int64
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    KVCacheEventManager as NativeKVCacheEventManager,
)

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import CudaStream, KVCacheManager, TokenId
    from kv_cache_manager_v2._block_radix_tree import ReuseScope
    from kv_cache_manager_v2._utils import TemporaryCudaStream, init_cuda_once, temporary_sys_path
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CudaStream, KVCacheManager, TokenId
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import ReuseScope
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        TemporaryCudaStream,
        init_cuda_once,
        temporary_sys_path,
    )

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from test_kv_cache_manager_v2 import create_config

# The probe lives in the pyexecutor wrapper; the key math is shared so it can be
# exercised directly against the core manager.
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (  # noqa: E402  isort:skip
    _first_new_block_key,
)

TOKENS_PER_BLOCK = 4


class TestFirstNewBlockProbe(unittest.TestCase):
    """Uniform full attention unless a test opts into a windowed layout."""

    # Even layers take ``window_size``, odd layers stay full attention (see
    # create_config), so a non-None value here yields two distinct life cycles.
    window_size = None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.event_manager = NativeKVCacheEventManager(max_kv_event_entries=1024)
        self.manager = KVCacheManager(
            create_config(
                tokens_per_block=TOKENS_PER_BLOCK,
                gpu_quota=16 << 20,
                host_quota=0,
                disk_quota=0,
                num_layers=2,
                window_size=self.window_size,
                sink_tokens=0,
            ),
            event_manager=self.event_manager,
        )

    def tearDown(self) -> None:
        gc.enable()
        if hasattr(self, "manager"):
            self.manager.shutdown()
            del self.manager

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def tokens(start, count):
        return [TokenId(start + i) for i in range(count)]

    def probe_key(self, tokens, reuse_scope=None):
        """What the scheduler's probe would return for *tokens*."""
        scope = ReuseScope() if reuse_scope is None else reuse_scope
        num_reusable = self.manager.probe_reuse(scope, tokens)
        return _first_new_block_key(tokens, TOKENS_PER_BLOCK, scope, num_reusable)

    def commit(self, tokens, reuse_scope=None):
        scope = ReuseScope() if reuse_scope is None else reuse_scope
        with TemporaryCudaStream([]) as stream_holder:
            stream = cast(CudaStream, stream_holder.handle)
            kv_cache = self.manager.create_kv_cache(scope, tokens)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(len(tokens)))
            uncommitted = tokens[kv_cache.num_committed_tokens :]
            if uncommitted:
                kv_cache.commit(uncommitted)
            kv_cache.close()
        stream_holder.take_finish_event().synchronize()

    def stored_block_hashes(self):
        self.event_manager.flush_iteration_events()
        events = KVCacheEventSerializer.serialize(self.event_manager.get_latest_events(0))
        return [
            block["block_hash"]
            for event in events
            if event["data"]["type"] == "stored"
            for block in event["data"]["blocks"]
        ]

    @staticmethod
    def hash_candidates(key):
        """Event ``block_hash`` forms a raw key can normalize to.

        The event manager emits either the hex digest or a truncated int64
        depending on the configured hash algorithm; accept either so the test
        does not pin an unrelated setting.
        """
        return {key.hex(), truncate_sha256_hash_to_int64(key)}

    # ---- tests ------------------------------------------------------------

    def test_probe_names_the_block_that_gets_stored(self):
        """The probed key is the key of the first block the commit stores."""
        tokens = self.tokens(0, 4 * TOKENS_PER_BLOCK)
        key = self.probe_key(tokens)
        self.assertIsNotNone(key)
        self.stored_block_hashes()  # drain setup events
        self.commit(tokens)
        stored = self.stored_block_hashes()
        self.assertTrue(stored, "commit produced no stored-block events")
        self.assertIn(stored[0], self.hash_candidates(key))

    def test_duplicates_collide_then_diverge_after_commit(self):
        """Duplicates collide, then diverge once one of them commits.

        This is the property the scheduler acts on: two requests with the same
        uncached prefix probe to the same key -- so deferring one is useful --
        and they stop colliding after the other has committed.
        """
        shared = self.tokens(0, 3 * TOKENS_PER_BLOCK)
        first = shared + self.tokens(100, TOKENS_PER_BLOCK)
        second = shared + self.tokens(200, TOKENS_PER_BLOCK)

        first_key = self.probe_key(first)
        second_key = self.probe_key(second)
        self.assertIsNotNone(first_key)
        self.assertEqual(first_key, second_key)

        self.commit(first)

        # The shared prefix is cached now, so the duplicate no longer starts at
        # the same block -- exactly the recomputation the deferral avoided.
        advanced_key = self.probe_key(second)
        self.assertIsNotNone(advanced_key)
        self.assertNotEqual(advanced_key, second_key)
        self.assertGreaterEqual(self.manager.probe_reuse(ReuseScope(), second), len(shared))

    def test_none_when_next_block_would_be_partial(self):
        """Mirrors v1's nullopt: nothing to register until a block is full."""
        self.assertIsNone(self.probe_key(self.tokens(0, TOKENS_PER_BLOCK - 1)))
        tokens = self.tokens(0, 2 * TOKENS_PER_BLOCK)
        self.commit(tokens)
        # Fully cached prompt: the request contributes no new full block.
        self.assertIsNone(self.probe_key(tokens))

    def test_distinct_prefixes_do_not_collide(self):
        self.assertNotEqual(
            self.probe_key(self.tokens(0, 2 * TOKENS_PER_BLOCK)),
            self.probe_key(self.tokens(500, 2 * TOKENS_PER_BLOCK)),
        )

    def test_reuse_scope_separates_keys(self):
        tokens = self.tokens(0, 2 * TOKENS_PER_BLOCK)
        self.assertNotEqual(
            self.probe_key(tokens, ReuseScope()),
            self.probe_key(tokens, ReuseScope(salt=1234)),
        )


class TestFirstNewBlockProbeVswa(TestFirstNewBlockProbe):
    """Same contract on a variable-window layout.

    ``window_size`` shorter than the shared prefix is the interesting case: the
    contributor's sliding-window pages for early blocks are released at commit,
    yet those blocks are also the ones ``get_stale_range`` marks not-required,
    so the deferred duplicate still matches past them.
    """

    window_size = TOKENS_PER_BLOCK
