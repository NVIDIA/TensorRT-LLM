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
"""Unit tests for KV cache salting in the V2 manager.

Verifies that ``cache_salt_id`` correctly isolates radix-tree branches:
- Same salt → reuse works
- Different salts → no reuse
- Salted vs non-salted → no reuse
- No salt on either side → reuse works (backward compatibility)
- ``_make_tree_task_id`` is deterministic and distinguishes distinct salts.
"""

import gc
import os
import unittest
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import CudaStream, KVCacheManager
    from kv_cache_manager_v2._core._kv_cache import _KVCache
    from kv_cache_manager_v2._utils import TemporaryCudaStream, init_cuda_once, temporary_sys_path
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CudaStream, KVCacheManager
    from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache import _KVCache
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        TemporaryCudaStream,
        init_cuda_once,
        temporary_sys_path,
    )

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from test_kv_cache_manager_v2 import create_config


def _make_manager() -> KVCacheManager:
    """Build a small single-layer manager suitable for reuse tests."""
    cfg = create_config(
        tokens_per_block=8,
        gpu_quota=16 << 20,
        host_quota=0,
        disk_quota=0,
        num_layers=2,
        window_size=None,
        sink_tokens=0,
        kv_buf_size=8192,
    )
    return KVCacheManager(cfg)


class TestKVCacheSaltedReuse(unittest.TestCase):
    """Salting must isolate reuse across requests with mismatched salts."""

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
        """Resume, resize, commit all tokens, stop committing, close."""
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(capacity))
            uncommitted = tokens[kv_cache.num_committed_tokens :]
            if uncommitted:
                kv_cache.commit(uncommitted)
            kv_cache.stop_committing()
        kv_cache.close()

    def test_same_salt_reuses_blocks(self) -> None:
        """Two requests with the same ``cache_salt_id`` must reuse blocks."""
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128
        salt = 12345

        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=salt)
        self._commit_and_close(kv_a, tokens, capacity)

        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=salt)
        self.assertGreater(kv_b.num_committed_tokens, 0, "Same salt must reuse committed blocks")
        self._commit_and_close(kv_b, tokens, capacity)

    def test_different_salt_does_not_reuse_blocks(self) -> None:
        """Same tokens but different ``cache_salt_id`` must NOT reuse blocks."""
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128

        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=12345)
        self._commit_and_close(kv_a, tokens, capacity)

        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=67890)
        self.assertEqual(
            kv_b.num_committed_tokens,
            0,
            "Different salt must not reuse blocks committed under another salt",
        )
        kv_b.close()

    def test_salted_vs_non_salted_does_not_reuse(self) -> None:
        """A salted request must not reuse blocks committed by a non-salted one (and vice versa)."""
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128

        # Non-salted commit first.
        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1])
        self._commit_and_close(kv_a, tokens, capacity)

        # Salted request must NOT see those blocks.
        kv_salted = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=42)
        self.assertEqual(kv_salted.num_committed_tokens, 0)
        self._commit_and_close(kv_salted, tokens, capacity)

        # And a fresh non-salted request must NOT see the salted blocks either,
        # but it CAN still see the original non-salted blocks committed earlier.
        kv_plain = self.manager.create_kv_cache(input_tokens=tokens[:-1])
        self.assertGreater(
            kv_plain.num_committed_tokens,
            0,
            "Non-salted reuse of existing non-salted blocks must still work",
        )
        kv_plain.close()

    def test_no_salt_backward_compat(self) -> None:
        """Two requests without salt must reuse blocks the same as before."""
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128

        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1])
        self._commit_and_close(kv_a, tokens, capacity)

        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1])
        self.assertGreater(
            kv_b.num_committed_tokens, 0, "Non-salted requests must reuse like before"
        )
        self._commit_and_close(kv_b, tokens, capacity)

    def test_independent_branches_per_salt(self) -> None:
        """Different salts must each independently build and reuse their own blocks."""
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128

        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=1)
        self._commit_and_close(kv_a, tokens, capacity)

        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=2)
        self.assertEqual(kv_b.num_committed_tokens, 0, "First request on salt=2 sees nothing yet")
        self._commit_and_close(kv_b, tokens, capacity)

        # Both salts now have committed blocks — each must reuse only its own.
        kv_a2 = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=1)
        self.assertGreater(kv_a2.num_committed_tokens, 0)
        self._commit_and_close(kv_a2, tokens, capacity)

        kv_b2 = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=2)
        self.assertGreater(kv_b2.num_committed_tokens, 0)
        self._commit_and_close(kv_b2, tokens, capacity)


class TestMakeTreeTaskId(unittest.TestCase):
    """Direct tests for ``_KVCache._make_tree_task_id``."""

    def test_no_salt_returns_lora_task_id_unchanged(self) -> None:
        """``cache_salt_id=None`` preserves pre-salting behavior exactly."""
        self.assertIsNone(_KVCache._make_tree_task_id(None, None))
        self.assertEqual(_KVCache._make_tree_task_id(7, None), 7)

    def test_distinct_salts_give_distinct_seeds(self) -> None:
        a = _KVCache._make_tree_task_id(None, 1)
        b = _KVCache._make_tree_task_id(None, 2)
        self.assertIsInstance(a, bytes)
        self.assertIsInstance(b, bytes)
        self.assertNotEqual(a, b)

    def test_salt_and_no_salt_give_distinct_seeds(self) -> None:
        """``cache_salt_id`` set vs unset must never collide for the same lora id."""
        plain = _KVCache._make_tree_task_id(42, None)
        salted = _KVCache._make_tree_task_id(42, 0)
        self.assertNotEqual(plain, salted)

    def test_deterministic(self) -> None:
        """Same inputs must produce byte-identical seeds."""
        seed_a = _KVCache._make_tree_task_id(42, 12345)
        seed_b = _KVCache._make_tree_task_id(42, 12345)
        self.assertEqual(seed_a, seed_b)

    def test_lora_id_part_of_seed(self) -> None:
        """Same salt with different lora ids must differ."""
        s1 = _KVCache._make_tree_task_id(1, 12345)
        s2 = _KVCache._make_tree_task_id(2, 12345)
        self.assertNotEqual(s1, s2)

    def test_rejects_negative_lora_task_id(self) -> None:
        """Negative lora_task_id would alias the ``None`` sentinel; must raise."""
        with self.assertRaises(ValueError):
            _KVCache._make_tree_task_id(-1, 12345)

    def test_rejects_negative_cache_salt_id(self) -> None:
        """cache_salt_id is unsigned 8B; negative must raise."""
        with self.assertRaises(ValueError):
            _KVCache._make_tree_task_id(0, -1)

    def test_rejects_overflowing_cache_salt_id(self) -> None:
        """cache_salt_id must fit in unsigned 8 bytes; overflow must raise."""
        with self.assertRaises(ValueError):
            _KVCache._make_tree_task_id(0, 1 << 64)


class TestLoraSaltInteraction(unittest.TestCase):
    """LoRA task ID and cache salt must compose: each pair gets its own tree branch.

    Goes through the public ``KVCacheManager.create_kv_cache`` API rather than
    real LoRA infrastructure, since the salt-isolation contract lives at the
    radix-tree level. Covers all four combinations of
    ``(lora_task_id, cache_salt_id)`` plus a cross-pair check.
    """

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

    def _commit_for(self, lora_task_id, cache_salt_id, tokens, capacity):
        kv = self.manager.create_kv_cache(
            lora_task_id=lora_task_id,
            input_tokens=tokens[:-1],
            cache_salt_id=cache_salt_id,
        )
        self._commit_and_close(kv, tokens, capacity)

    def _peek_committed(self, lora_task_id, cache_salt_id, tokens) -> int:
        """Open a fresh KV cache and return reusable token count without committing.

        Used to probe whether a particular (lora, salt) branch already has
        committed blocks visible to a new request.
        """
        kv = self.manager.create_kv_cache(
            lora_task_id=lora_task_id,
            input_tokens=tokens[:-1],
            cache_salt_id=cache_salt_id,
        )
        n = kv.num_committed_tokens
        kv.close()
        return n

    def test_all_four_combinations_isolated(self) -> None:
        """Each (lora_task_id, cache_salt_id) pair lands on its own radix-tree branch.

        Covers the four combinations (None,None), (L,None), (None,S), (L,S):
        commits in any one branch must not leak into the other three.
        """
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128

        L = 7
        S = 12345

        # Commit on (None, None) only — the other three branches must remain empty.
        self._commit_for(None, None, tokens, capacity)
        self.assertGreater(self._peek_committed(None, None, tokens), 0)
        self.assertEqual(self._peek_committed(L, None, tokens), 0)
        self.assertEqual(self._peek_committed(None, S, tokens), 0)
        self.assertEqual(self._peek_committed(L, S, tokens), 0)

        # Commit on (L, None) — that branch fills, others unchanged.
        self._commit_for(L, None, tokens, capacity)
        self.assertGreater(self._peek_committed(L, None, tokens), 0)
        self.assertEqual(self._peek_committed(None, S, tokens), 0)
        self.assertEqual(self._peek_committed(L, S, tokens), 0)

        # Commit on (None, S) — that branch fills, (L, S) still empty.
        self._commit_for(None, S, tokens, capacity)
        self.assertGreater(self._peek_committed(None, S, tokens), 0)
        self.assertEqual(self._peek_committed(L, S, tokens), 0)

        # Commit on (L, S) — finally fills.
        self._commit_for(L, S, tokens, capacity)
        self.assertGreater(self._peek_committed(L, S, tokens), 0)

    def test_same_lora_different_salt_isolated(self) -> None:
        """Two requests with the same LoRA but different salts must NOT reuse."""
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128
        lora = 99

        self._commit_for(lora, 1111, tokens, capacity)
        self.assertEqual(self._peek_committed(lora, 2222, tokens), 0)
        self.assertGreater(self._peek_committed(lora, 1111, tokens), 0)

    def test_same_salt_different_lora_isolated(self) -> None:
        """Two requests with the same salt but different LoRA ids must NOT reuse."""
        self.manager = _make_manager()
        tokens = list(range(64))
        capacity = 128
        salt = 9999

        self._commit_for(1, salt, tokens, capacity)
        self.assertEqual(self._peek_committed(2, salt, tokens), 0)
        self.assertGreater(self._peek_committed(1, salt, tokens), 0)


class TestKVCacheSaltCounters(unittest.TestCase):
    """Aggregate manager counters must reflect creation/close events deterministically.

    These are the counters listed in
    ``tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache_manager.py`` slots
    (``_num_created_kv_caches``, ``_num_closed_kv_caches``, ``_avg_reused_length``,
    ``_avg_sqr_capacity``, ``_avg_sqr_history_length``). Salting changes which
    blocks are eligible for reuse, but the counter mechanics are unchanged. We
    pick a tightly controlled scenario (16 tokens = 2 full blocks of size 8,
    single salt, full reuse on the second request) so every counter value can
    be derived analytically.
    """

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

    def test_aggregate_counters_match_expected(self) -> None:
        """Two committed-then-closed requests with full reuse: every counter has an exact value.

        Scenario (tokens_per_block=8, full attention only):
          R1: empty tree, ``input_tokens=[0..15]`` (16 tokens, exactly 2 full blocks).
              No reuse. Cache._avg_capacity sees update(0) at init, update(16) at
              resize → cache average 8. Cache._avg_history_length sees update(0)
              at init only → cache average 0. close() pushes 8**2=64 into
              manager._avg_sqr_capacity and 0**2=0 into _avg_sqr_history_length;
              create-time pushes history_length=0 into _avg_reused_length.
          R2: tree fully populated with R1's blocks under the same salt, full
              reuse → _setup_for_reuse sets _history_length = _capacity = 16
              before the cache-level averages are constructed. Cache._avg_capacity
              sees update(16) twice (init + resize) → cache average 16.
              Cache._avg_history_length sees update(16) once (init) → 16.
              close() pushes 16**2=256 into both _avg_sqr_* counters; create-time
              pushes 16 into _avg_reused_length.

        ``MovingAverage(decay=0.9999)`` formula (see _moving_average.py:30-34):
          weight_n = 1 + decay * weight_(n-1);  avg_n = avg_(n-1) + (v - avg_(n-1)) / weight_n.
        """
        cfg = create_config(
            tokens_per_block=8,
            gpu_quota=16 << 20,
            host_quota=0,
            disk_quota=0,
            num_layers=2,
            window_size=None,
            sink_tokens=0,
            kv_buf_size=8192,
        )
        self.manager = KVCacheManager(cfg)
        tokens = list(range(16))  # 16 tokens, exactly 2 full blocks of size 8.
        salt = 4242

        # R1: no reuse possible (empty tree).
        kv_a = self.manager.create_kv_cache(input_tokens=tokens, cache_salt_id=salt)
        self.assertEqual(kv_a.num_committed_tokens, 0)
        self._commit_and_close(kv_a, tokens, capacity=16)

        # R2: full reuse against R1's committed blocks (same salt, same tokens).
        kv_b = self.manager.create_kv_cache(input_tokens=tokens, cache_salt_id=salt)
        self.assertEqual(kv_b.num_committed_tokens, 16, "Expected full reuse on second request")
        self._commit_and_close(kv_b, tokens, capacity=16)

        # 2 create_kv_cache calls → 2 created.
        self.assertEqual(self.manager._num_created_kv_caches, 2)
        # 2 close() calls → 2 closed.
        self.assertEqual(self.manager._num_closed_kv_caches, 2)

        # _avg_reused_length: MovingAverage updated with 0, then 16.
        #   step 1: weight=1.0,         avg = 0.0
        #   step 2: weight=1.9999,      avg = 0 + (16 - 0)/1.9999 = 8.000400020001
        # round(8.000400020001) == 8.
        self.assertEqual(round(self.manager._avg_reused_length.value), 8)

        # _avg_sqr_capacity: MovingAverage updated with 64 (=8^2), then 256 (=16^2).
        #   step 1: weight=1.0,         avg = 64.0
        #   step 2: weight=1.9999,      avg = 64 + (256 - 64)/1.9999 = 160.004800240012
        self.assertAlmostEqual(self.manager._avg_sqr_capacity.value, 160.004800240012, places=6)

        # _avg_sqr_history_length: MovingAverage updated with 0, then 256 (=16^2).
        #   step 1: weight=1.0,         avg = 0.0
        #   step 2: weight=1.9999,      avg = 0 + (256 - 0)/1.9999 = 128.006400320016
        self.assertAlmostEqual(
            self.manager._avg_sqr_history_length.value, 128.006400320016, places=6
        )


def _make_vswa_manager() -> KVCacheManager:
    """Manager with mixed sliding-window/full-attention layers (VSWA-like).

    ``create_config`` puts ``sliding_window_size=window_size`` on even layers
    and ``None`` on odd layers, so passing ``num_layers=4, window_size=32``
    yields ``[swa(32), full, swa(32), full]`` — heterogeneous per-layer windows.
    """
    cfg = create_config(
        tokens_per_block=8,
        gpu_quota=16 << 20,
        host_quota=0,
        disk_quota=0,
        num_layers=4,
        window_size=32,
        sink_tokens=0,
        kv_buf_size=8192,
    )
    return KVCacheManager(cfg)


class TestKVCacheVSWASaltIsolation(unittest.TestCase):
    """Per-layer/per-window pools must each see independent salt branches.

    The radix tree keys all blocks (across every life cycle / pool group) with
    the same ``tree_task_id``, so salt isolation at the tree root implies
    isolation across every per-window pool downstream. We assert the
    cross-window-visible signal (``num_committed_tokens``) which is the API
    actually surfaced to users; per-pool-group hit counters are not part of
    the public manager API today.
    """

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

    def test_vswa_per_window_salt_isolation(self) -> None:
        """In a VSWA layout, two different salts on identical tokens must not reuse anything.

        With per-layer windows ``[32, None, 32, None]`` and tokens_per_block=8,
        a 32-token prompt spans 4 blocks. Even for the SWA layers (which only
        retain the in-window suffix), the radix-tree match is gated by salt at
        the root, so a request with a different salt sees zero reusable tokens
        regardless of which pool/window the underlying block was provisioned
        in.
        """
        self.manager = _make_vswa_manager()
        tokens = list(range(64))
        capacity = 128

        # First salt: commits.
        kv_a = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=1001)
        self._commit_and_close(kv_a, tokens, capacity)

        # Same salt sees the previously committed blocks.
        kv_a2 = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=1001)
        self.assertGreater(
            kv_a2.num_committed_tokens,
            0,
            "Same-salt request must reuse VSWA blocks committed earlier",
        )
        kv_a2.close()

        # Different salt: zero reuse, regardless of per-window pool layout.
        kv_b = self.manager.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=2002)
        self.assertEqual(
            kv_b.num_committed_tokens,
            0,
            "Cross-salt reuse must be zero across every per-window pool group",
        )
        kv_b.close()


class TestKVCacheDisaggSalt(unittest.TestCase):
    """Salt propagation through a disaggregated-serving-style two-manager handoff.

    The C++ disagg path passes ``cacheSaltID`` end-to-end. At the V2 Python
    unit-test layer there is no transfer machinery that automatically threads
    salt from a context manager to a generation manager — both managers each
    create their own ``_KVCache`` objects and the test harness controls whether
    salt is supplied. This test pins the **observed** behavior:

      * The receiver creates a fresh KV cache; if it passes ``cache_salt_id=A``
        and the sender used ``A``, reuse on the *receiver's own* manager only
        kicks in once that receiver has seen something committed under salt A
        (which the unit harness does not auto-populate via transfer).
      * Therefore at the unit-test layer the receiver's first
        ``num_committed_tokens`` is 0 regardless of the salt match — there's
        no shared radix tree across managers.

    This is structured as a regression test: it documents the silent-fallback
    behavior so a future PR that wires unit-level salt propagation across the
    transfer can flip the assertions intentionally rather than discover the
    change accidentally.
    """

    sender: KVCacheManager | None
    receiver: KVCacheManager | None

    def setUp(self) -> None:
        init_cuda_once()
        gc.collect()
        gc.disable()
        self.sender = None
        self.receiver = None

    def tearDown(self) -> None:
        gc.enable()
        if self.sender is not None:
            self.sender.shutdown()
            self.sender = None
        if self.receiver is not None:
            self.receiver.shutdown()
            self.receiver = None

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

    def test_disagg_path_with_salt(self) -> None:
        """Two-manager handoff: receiver does NOT inherit sender's reusable blocks.

        Models ``TestDisagg`` / ``TestDisaggregatedServing`` in
        ``test_kv_cache_manager_v2.py``: a context-side manager commits under
        salt A, and a separate generation-side manager creates a fresh KV
        cache. Even with matching salt, the receiver's radix tree is empty,
        so ``num_committed_tokens`` on the receiver's first request is 0.

        This is the unit-layer baseline. The C++ side propagates salt through
        the actual transfer (BlockKey::cacheSaltID); this Python unit test
        does not exercise that transfer machinery. Treat as a guard against
        accidental cross-manager radix-tree leakage.
        """
        cfg = create_config(
            tokens_per_block=8,
            gpu_quota=16 << 20,
            host_quota=0,
            disk_quota=0,
            num_layers=2,
            window_size=None,
            sink_tokens=0,
            kv_buf_size=8192,
        )
        self.sender = KVCacheManager(cfg)
        self.receiver = KVCacheManager(cfg)

        tokens = list(range(64))
        capacity = 128
        salt = 31415

        # Sender: commit under salt A.
        kv_send = self.sender.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=salt)
        self._commit_and_close(kv_send, tokens, capacity)

        # Receiver: same salt A, no transfer — fresh radix tree.
        kv_recv_same = self.receiver.create_kv_cache(input_tokens=tokens[:-1], cache_salt_id=salt)
        self.assertEqual(
            kv_recv_same.num_committed_tokens,
            0,
            "Receiver must not see sender-committed blocks without an explicit transfer",
        )
        kv_recv_same.close()

        # Sanity: even on the sender's own manager, salt isolation still holds.
        kv_send_other_salt = self.sender.create_kv_cache(
            input_tokens=tokens[:-1], cache_salt_id=salt + 1
        )
        self.assertEqual(
            kv_send_other_salt.num_committed_tokens,
            0,
            "Different salt must not reuse blocks even on the same manager",
        )
        kv_send_other_salt.close()


class TestLoraSaltBoundaryValues(unittest.TestCase):
    """Boundary checks around ``_make_tree_task_id`` lora/salt range guards.

    Complements ``TestMakeTreeTaskId`` above (which already covers negatives,
    overflow on both sides, and basic distinctness). Focuses on the exact
    accepted-vs-rejected boundary and on the ``cache_salt_id=0`` edge.
    """

    def test_max_signed_lora_task_id_accepted(self) -> None:
        """``lora_task_id == (1<<63) - 1`` is well within the accepted range.

        ``_make_tree_task_id`` rejects ``not 0 <= x < (1 << 64)`` for
        ``lora_task_id``, so ``(1<<63) - 1`` must pass. Use a non-None
        ``cache_salt_id`` so the validation path runs (the no-salt fast-path
        bypasses range checks).
        """
        seed = _KVCache._make_tree_task_id((1 << 63) - 1, 12345)
        self.assertIsInstance(seed, bytes)

    def test_lora_task_id_at_signed_overflow_accepted(self) -> None:
        """``lora_task_id == 1 << 63`` must be accepted under the unsigned-8B bound.

        Earlier the bound was signed (``< 1 << 63``); the field is now treated
        as unsigned 8 bytes so values in ``[1<<63, 1<<64)`` are valid IDs.
        """
        seed = _KVCache._make_tree_task_id(1 << 63, 12345)
        self.assertIsInstance(seed, bytes)

    def test_max_unsigned_lora_task_id_accepted(self) -> None:
        """``lora_task_id == (1<<64) - 1`` is the largest accepted unsigned-8B value."""
        seed = _KVCache._make_tree_task_id((1 << 64) - 1, 12345)
        self.assertIsInstance(seed, bytes)

    def test_max_unsigned_cache_salt_id_accepted(self) -> None:
        """``cache_salt_id == (1<<64) - 1`` is the largest accepted unsigned-8B value."""
        seed = _KVCache._make_tree_task_id(0, (1 << 64) - 1)
        self.assertIsInstance(seed, bytes)

    def test_zero_salt_distinct_from_no_salt(self) -> None:
        """``cache_salt_id=0`` is a real salt and must not collide with ``None``.

        Important because the no-salt fast-path returns ``lora_task_id`` as an
        ``int`` while the salted path returns a ``bytes`` digest — even at
        ``salt=0`` the two values must be distinguishable for radix-tree
        branch isolation.
        """
        plain = _KVCache._make_tree_task_id(0, None)
        salted_zero = _KVCache._make_tree_task_id(0, 0)
        self.assertNotEqual(plain, salted_zero)
        self.assertNotIsInstance(plain, bytes)
        self.assertIsInstance(salted_zero, bytes)

    def test_none_lora_distinct_from_zero_lora_under_same_salt(self) -> None:
        r"""``(None, S)`` and ``(0, S)`` must yield distinct seeds.

        Both inputs go through the salted path so both return ``bytes``
        digests; the domain-separation byte (``\\x00`` for ``None``,
        ``\\x01`` for set) is the only thing distinguishing them. Without
        that byte the two digests would collide and a no-LoRA salted request
        would share a radix-tree branch with a ``lora_task_id=0`` salted
        request.
        """
        no_lora = _KVCache._make_tree_task_id(None, 12345)
        zero_lora = _KVCache._make_tree_task_id(0, 12345)
        self.assertIsInstance(no_lora, bytes)
        self.assertIsInstance(zero_lora, bytes)
        self.assertNotEqual(no_lora, zero_lora)

    def test_digest_regression_pin(self) -> None:
        r"""Pin exact digests so a future hash-format change is caught.

        The radix-tree seed governs which committed blocks a request can
        reuse. A silent change to digest derivation (added/removed input
        bytes, endianness flip, different separator) would invalidate every
        previously committed branch under the same ``(lora_task_id,
        cache_salt_id)`` pair. Two representative inputs cover both code
        paths: ``lora`` set (``\\x01`` prefix) and ``lora=None`` (``\\x00``
        prefix).
        """
        self.assertEqual(
            _KVCache._make_tree_task_id(42, 12345).hex(),
            "8ba1613eb10f7f741a6f363ac69a35b296ce7a8ca56a6e7a5018516655c0dd03",
        )
        self.assertEqual(
            _KVCache._make_tree_task_id(None, 12345).hex(),
            "a3b5cd3e2949921c7ad0e3ec8a9401293cd2812951ffe9a360a1d59270ff9d86",
        )


class TestMultimodalAugmentedSalt(unittest.TestCase):
    """Multimodal augmentation must compose with salt at the manager-unit layer.

    The production path is in ``resource_manager.py`` (``_create_kv_cache``
    callsite around line 2291): tokens are augmented via
    ``_augment_tokens_for_block_reuse`` before being passed to
    ``KVCacheManager.create_kv_cache(..., cache_salt_id=req.cache_salt_id)``.
    Here we don't go through ``LlmRequest`` plumbing — we directly feed
    augmented-style token sequences (where one token slot is replaced with the
    multimodal content digest, mirroring ``gen_multi_modal_tokens``) into
    ``create_kv_cache`` and assert salt-driven isolation downstream of the
    augmentation. This keeps the test focused on the manager-layer composition
    contract instead of the resource-manager wiring.
    """

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

    def _commit_and_close(self, kv_cache: _KVCache, tokens: list, capacity: int) -> None:
        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)
            self.assertTrue(kv_cache.resume(stream))
            self.assertTrue(kv_cache.resize(capacity))
            uncommitted = tokens[kv_cache.num_committed_tokens :]
            if uncommitted:
                kv_cache.commit(uncommitted)
            kv_cache.stop_committing()
        kv_cache.close()

    def test_multimodal_augmented_path_composes_with_salt(self) -> None:
        """Two requests with identical augmented tokens but different salts must NOT reuse.

        The augmented sequence here mirrors ``gen_multi_modal_tokens``: a
        32-byte digest token in slot 0, then plain-int tokens after. Identical
        digest plus identical tail tokens means identical input token streams
        — so any reuse decision is driven entirely by the salt.
        """
        cfg = create_config(
            tokens_per_block=8,
            gpu_quota=16 << 20,
            host_quota=0,
            disk_quota=0,
            num_layers=2,
            window_size=None,
            sink_tokens=0,
            kv_buf_size=8192,
        )
        self.manager = KVCacheManager(cfg)

        # Identical "augmented" token sequence for both requests: a 32-byte
        # multimodal digest followed by trailing int tokens. Same content →
        # same hash chain → reuse decision is determined by salt alone.
        digest = b"\xa5" * 32
        augmented = [digest] + list(range(1, 64))
        capacity = 128

        kv_a = self.manager.create_kv_cache(input_tokens=augmented[:-1], cache_salt_id=111)
        self._commit_and_close(kv_a, augmented, capacity)

        kv_b = self.manager.create_kv_cache(input_tokens=augmented[:-1], cache_salt_id=222)
        self.assertEqual(
            kv_b.num_committed_tokens,
            0,
            "Identical multimodal-augmented tokens with different salts must NOT reuse",
        )
        kv_b.close()

        # Sanity: same salt + same augmented tokens → reuse works.
        kv_c = self.manager.create_kv_cache(input_tokens=augmented[:-1], cache_salt_id=111)
        self.assertGreater(
            kv_c.num_committed_tokens,
            0,
            "Same salt + same augmented tokens must reuse the prior commit",
        )
        kv_c.close()


if __name__ == "__main__":
    unittest.main()
