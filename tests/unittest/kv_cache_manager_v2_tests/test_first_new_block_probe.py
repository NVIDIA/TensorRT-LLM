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

``KVCacheManager.probe_first_new_block_key``
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
import sys
import unittest
from collections.abc import Sequence
from dataclasses import replace
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np

from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.runtime.kv_cache_hash import truncate_sha256_hash_to_int64
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    KVCacheEventManager as NativeKVCacheEventManager,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import sequence_to_blockchain_keys

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import CudaStream, KVCacheManager, ReuseScope, TokenId, _introspection
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        CudaStream,
        KVCacheManager,
        ReuseScope,
        TokenId,
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
        TemporaryCudaStream,
        init_cuda_once,
        temporary_sys_path,
    )
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)

with temporary_sys_path(_TEST_DIR):
    from test_kv_cache_manager_v2 import create_config

TOKENS_PER_BLOCK = 4


def reference_key(
    tokens: Sequence[int | bytes] | np.ndarray, scope: ReuseScope, num_reusable: int
) -> bytes | None:
    """Select the first unfinished full block from the existing full-chain iterator."""
    for ordinal, (block, key) in enumerate(
        sequence_to_blockchain_keys(TOKENS_PER_BLOCK, scope, list(tokens))
    ):
        if len(block) == TOKENS_PER_BLOCK and ordinal * TOKENS_PER_BLOCK > num_reusable:
            return key
    return None


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
        key = self.manager.probe_first_new_block_key(scope, tokens)
        self.assertEqual(key, reference_key(tokens, scope, num_reusable))
        return key

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
        self.assertIsNone(self.probe_key([]))
        self.assertIsNone(self.probe_key(self.tokens(0, TOKENS_PER_BLOCK - 1)))
        tokens = self.tokens(0, 2 * TOKENS_PER_BLOCK)
        self.commit(tokens)
        # Fully cached prompt: the request contributes no new full block.
        self.assertIsNone(self.probe_key(tokens))
        # An uncached partial tail still contributes no complete block.
        self.assertIsNone(self.probe_key(tokens + self.tokens(100, TOKENS_PER_BLOCK - 1)))

    def test_distinct_prefixes_do_not_collide(self):
        self.assertNotEqual(
            self.probe_key(self.tokens(0, 2 * TOKENS_PER_BLOCK)),
            self.probe_key(self.tokens(500, 2 * TOKENS_PER_BLOCK)),
        )

    def test_reuse_scope_separates_keys(self):
        tokens = self.tokens(0, 2 * TOKENS_PER_BLOCK)
        scopes = [
            ReuseScope(),
            ReuseScope(lora_id=0),
            ReuseScope(salt=0),
            ReuseScope(lora_id=0, salt=0),
            ReuseScope(lora_id=7, salt=(1 << 64) - 1),
        ]
        keys = [self.probe_key(tokens, scope) for scope in scopes]
        self.assertEqual(len(set(keys)), len(scopes))
        self.commit(tokens, scopes[-1])
        self.assertIsNone(self.probe_key(tokens, scopes[-1]))
        self.assertEqual(self.probe_key(tokens, scopes[0]), keys[0])

    def test_partial_match_hashes_the_query_suffix(self) -> None:
        tokens = self.tokens(0, 4 * TOKENS_PER_BLOCK)
        self.commit(tokens)
        for shared_length in (0, 1, 3, 4, 5, 7, 8, 11):
            with self.subTest(shared_length=shared_length):
                query = tokens[:shared_length] + self.tokens(100, len(tokens) - shared_length)
                # A partial tree block contains a different suffix, so using
                # its key (rather than its complete predecessor) is incorrect.
                if self.window_size is None:
                    self.assertEqual(self.manager.probe_reuse(None, query), shared_length)
                self.assertIsNotNone(self.probe_key(query))

    def test_backoff_uses_the_last_complete_matched_block(self) -> None:
        tokens = self.tokens(0, 4 * TOKENS_PER_BLOCK)
        scope = ReuseScope(lora_id=7, salt=1234)
        for backoff in (1, TOKENS_PER_BLOCK, TOKENS_PER_BLOCK + 1, len(tokens)):
            with self.subTest(backoff=backoff):
                config = replace(self.manager.init_config, reuse_match_backoff=backoff)
                self.manager.shutdown()
                self.manager = KVCacheManager(config, event_manager=self.event_manager)
                self.commit(tokens, scope)
                if self.window_size is None:
                    self.assertEqual(self.manager.probe_reuse(scope, tokens), len(tokens) - backoff)
                # The reusable endpoint can be inside a block, on a boundary,
                # or at the root. Compare with the complete hash-chain oracle.
                self.assertIsNotNone(self.probe_key(tokens, scope))

    def test_tree_removal_changes_the_probe(self) -> None:
        tokens = self.tokens(0, 3 * TOKENS_PER_BLOCK)
        cold_key = self.probe_key(tokens)
        self.commit(tokens[: 2 * TOKENS_PER_BLOCK])
        self.assertNotEqual(self.probe_key(tokens), cold_key)
        self.manager.clear_reusable_blocks()
        self.assertEqual(self.probe_key(tokens), cold_key)

    def test_digest_tokens_in_the_prefix_and_new_block(self) -> None:
        tokens = self.tokens(0, 4 * TOKENS_PER_BLOCK)
        tokens[3] = bytes([17]) * 32
        tokens[9] = bytes([23]) * 32
        scope = ReuseScope(lora_id=9, salt=4567)
        self.commit(tokens[: 2 * TOKENS_PER_BLOCK], scope)
        self.assertIsNotNone(self.probe_key(tokens, scope))
        self.assertNotEqual(self.probe_key(tokens, scope), self.probe_key(tokens, ReuseScope()))

    def test_int32_view_matches_list(self) -> None:
        tokens = self.tokens(0, 4 * TOKENS_PER_BLOCK)
        view = np.asarray(tokens, dtype=np.int32)
        view.flags.writeable = False
        strided = np.empty(len(tokens) * 2, dtype=np.int32)
        strided[::2] = view
        variants = (view, strided[::2], view.astype(np.int64))
        for cached_length in (0, 2 * TOKENS_PER_BLOCK):
            if cached_length:
                self.commit(tokens[:cached_length])
            for variant in variants:
                with self.subTest(cached_length=cached_length, strides=variant.strides):
                    self.assertEqual(self.probe_key(variant), self.probe_key(tokens))

    def test_array_probe_tracks_commits_and_mutation(self) -> None:
        """Array inputs name actual stored blocks as input and tree reuse change."""
        tokens = np.arange(4 * TOKENS_PER_BLOCK, dtype=np.int32)
        key = self.probe_key(tokens)
        self.assertIsNotNone(key)
        self.stored_block_hashes()
        self.commit(tokens.tolist())
        self.assertIn(self.stored_block_hashes()[0], self.hash_candidates(key))
        self.assertIsNone(self.probe_key(tokens))

        tokens[-1] += 100
        updated = self.probe_key(tokens)
        self.assertIsNotNone(updated)
        self.assertNotEqual(updated, key)
        self.commit(tokens.tolist())
        self.assertIn(self.stored_block_hashes()[0], self.hash_candidates(updated))

    def test_probe_preserves_storage_and_page_ownership(self) -> None:
        tokens = self.tokens(0, 4 * TOKENS_PER_BLOCK)
        self.commit(tokens[: 2 * TOKENS_PER_BLOCK])
        self.stored_block_hashes()

        def storage_snapshot() -> list[tuple[int, int, int]]:
            return [
                (stat.total, stat.free, stat.evictable)
                for stat in self.manager.get_storage_statistics()
            ]

        before = storage_snapshot()
        self.assertTrue(_introspection.all_tree_pages_droppable(self.manager))
        matched = self.manager.probe_reuse(None, tokens)
        expected = self.probe_key(tokens)
        for _ in range(3):
            self.assertEqual(self.manager.probe_first_new_block_key(None, tokens), expected)
        self.assertEqual(storage_snapshot(), before)
        self.assertTrue(_introspection.all_tree_pages_droppable(self.manager))
        self.assertEqual(self.manager.probe_reuse(None, tokens), matched)
        self.event_manager.flush_iteration_events()
        self.assertEqual(self.event_manager.get_latest_events(0), [])


class TestFirstNewBlockProbeVswa(TestFirstNewBlockProbe):
    """Same contract on a variable-window layout.

    ``window_size`` shorter than the shared prefix is the interesting case: the
    contributor's sliding-window pages for early blocks are released at commit,
    yet those blocks are also the ones ``get_stale_range`` marks not-required,
    so the deferred duplicate still matches past them.
    """

    window_size = TOKENS_PER_BLOCK

    def test_window_pruning_precedes_key_selection(self) -> None:
        tokens = self.tokens(0, 4 * TOKENS_PER_BLOCK)
        query = tokens[: 2 * TOKENS_PER_BLOCK] + self.tokens(100, 2 * TOKENS_PER_BLOCK)
        coverage = [TOKENS_PER_BLOCK] * len(_introspection.attention_life_cycle_ids(self.manager))
        first = _introspection.make_test_block(self.manager, tokens[:TOKENS_PER_BLOCK], coverage)
        blocks = [first]
        try:
            # Keep the second tree node and its full-attention page, but omit
            # its SWA page as happens after eviction. A normal commit alone
            # need not evict it, so construct this page-coverage state directly.
            for life_cycle in _introspection.swa_life_cycle_ids(self.manager):
                coverage[life_cycle] = 0
            second = _introspection.make_test_block(
                self.manager, tokens[TOKENS_PER_BLOCK : 2 * TOKENS_PER_BLOCK], coverage, first
            )
            blocks.append(second)
            self.assertEqual(self.manager.probe_reuse(None, query), TOKENS_PER_BLOCK)
            # The first block needing new pages already has a tree key: an
            # absent-key-only lookup would incorrectly advance past it.
            self.assertEqual(self.probe_key(query), _introspection.test_block_key(second))
        finally:
            for block in reversed(blocks):
                _introspection.close_test_block(block)
