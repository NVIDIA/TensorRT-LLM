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
"""Regression tests for https://nvbugs/6625710 (see also https://nvbugs/6553427).

An SSM snapshot is installed into the radix tree and immediately
``scheduleForEviction()``-ed, so the eviction policy is its *only* owner -- no
``KvCache`` ever holds or locks it.  The block it lives on is still referenced
by the producing request through ``SeqBlock::treeBlock``.

``Block::clearStaleBlocksAfterPageUnlink()`` prunes empty tail nodes with

    while (curr && curr->next.empty() && curr->storage.at(lcIdx) == nullptr)

so when the evicted page is the SSM snapshot the condition consults *only* the
SSM slot, and the block is detached even though a live request still references
it.  That request is left holding a tree block whose ``prev`` is null; its next
``_commitBlock()`` dereferences the stale parent link and dies in
``Block::tokensPerBlock()``.

The block is vulnerable only while it is still a *leaf* -- once the request
commits the following block ``next.empty()`` is false and the walk stops.  In
production that window is short, hence the ~1.7% failure rate.  These tests
hold it open by suspending the request; suspension is not part of the
mechanism, it only makes the race deterministic.

Two variants, because the bug takes two fixes:

* ``hybrid``   -- attention + SSM life cycles, as in Qwen3.5-35B-A3B, where the bug
  was first seen.  Covered by requiring *every* life-cycle slot to be empty before
  pruning a tail node: the attention page keeps a slot non-null, so the walk stops
  and the block is never detached.
* ``pure_ssm`` -- SSM life cycles only.  With no attention life cycle that
  all-life-cycles condition is trivially true and the block is detached exactly as
  before, so this variant additionally requires re-attaching the blocks a live
  request still holds.
"""

import gc
import itertools
import unittest
from typing import cast

# Import the module rather than `from ... import TestSSMSupport`: binding a TestCase
# subclass into this module's namespace makes pytest collect and re-run that entire
# sibling class here as well.
import test_kv_cache_manager_v2 as kv_test  # type: ignore[import-not-found]
from test_kv_cache_manager_v2 import (  # type: ignore[import-not-found]
    CachedCudaStream,
    CudaStream,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManager,
    TokenId,
    TokenIdExt,
    init_cuda_once,
)

TOKENS_PER_BLOCK = 32
PROMPT_BLOCKS = 2
CHURN_REQUESTS = 100


class TestNvBug6625710(unittest.TestCase):
    """Evicting an unheld SSM snapshot must not detach a still-referenced block."""

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
        return TokenId(next(self._token_id_gen))

    def _run(self, num_attn_layers: int, two_tier: bool = False) -> None:
        # _make_ssm_config does not touch `self`; reuse it directly.
        cfg = kv_test.TestSSMSupport._make_ssm_config(
            self,
            tokens_per_block=TOKENS_PER_BLOCK,
            gpu_quota=4 << 20,
            num_attn_layers=num_attn_layers,
            num_ssm_layers=2,
        )
        if two_tier:
            # GPU pages migrate down to host, so the *host* tier is the last level --
            # that is where forceEvict() actually drops DROPPABLE pages.
            cfg.cache_tiers = [
                GpuCacheTierConfig(quota=8 << 20),
                HostCacheTierConfig(quota=8 << 20),
            ]
        self.manager = KVCacheManager(cfg)
        stream_holder = CachedCudaStream()
        stream = cast(CudaStream, stream_holder.handle)

        # --- Request A commits a prefix, snapshotting SSM state into the tree.
        # From here the snapshot is owned solely by the eviction policy.
        kv_a = self.manager.create_kv_cache()
        self.assertTrue(kv_a.resume(stream))
        prompt: list[TokenIdExt] = []
        for i in range(PROMPT_BLOCKS):
            kv_a.capacity = TOKENS_PER_BLOCK * (i + 1)
            chunk = [self.next_token() for _ in range(TOKENS_PER_BLOCK)]
            kv_a.commit(chunk)
            prompt += chunk

        # A sleeps, holding its tree blocks. Its tail block is still a leaf.
        kv_a.suspend()

        # --- "A lot of requests ran during its sleep": each commits snapshots of
        # its own and closes, so the pool must reclaim A's unheld tail snapshot.
        for _ in range(CHURN_REQUESTS):
            kv = self.manager.create_kv_cache()
            if not kv.resume(stream):
                kv.close()
                continue
            for i in range(4):
                kv.capacity = TOKENS_PER_BLOCK * (i + 1)
                kv.commit([self.next_token() for _ in range(TOKENS_PER_BLOCK)])
            kv.close()

        # --- Precondition guard. The bug needs A's snapshot to have actually been
        # evicted; if the churn above stopped triggering eviction (quota re-tuned,
        # CHURN_REQUESTS lowered, eviction policy changed) the rest of this test
        # would pass without exercising anything. With no attention life cycle,
        # prefix reuse is only possible via an SSM snapshot, so a probe that can
        # still reuse the whole prompt proves the snapshot survived. The hybrid
        # config has no equivalent probe: its attention pages serve a prefix match
        # whether or not the snapshot is gone.
        if num_attn_layers == 0:
            probe = self.manager.create_kv_cache(input_tokens=list(prompt))
            reusable = probe.num_committed_tokens
            probe.close()
            self.assertLess(
                reusable,
                len(prompt),
                "A's SSM snapshot was not evicted, so this test is not exercising "
                "https://nvbugs/6625710 -- retune gpu_quota/CHURN_REQUESTS",
            )

        # --- A wakes up (the pool is free again) and keeps generating on top of
        # its committed prefix. If its tail block was detached, this commit walks
        # a null `prev`.
        self.assertTrue(kv_a.resume(stream))
        kv_a.capacity = len(prompt) + TOKENS_PER_BLOCK
        kv_a.commit([self.next_token() for _ in range(TOKENS_PER_BLOCK)])

        kv_a.close()
        self.manager.shutdown()

    def test_hybrid_attention_and_ssm(self) -> None:
        self._run(num_attn_layers=2, two_tier=True)

    def test_pure_ssm(self) -> None:
        self._run(num_attn_layers=0)
