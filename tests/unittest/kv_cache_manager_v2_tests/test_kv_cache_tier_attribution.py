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
"""Storage-tier attribution of prefix-cache hits in KVCacheManagerV2.

A reused block is attributed to the cache level its page was on when the prefix
was matched, before any migration back to GPU. These tests drive the runtime
manager directly: commit a prompt, push its pages to the host tier by shrinking
the GPU quota, match the same prompt again and check both the per-request
``(tier, tokens)`` segments and the block-level statistics.
"""

import os
from typing import cast

import pytest
import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    CudaStream,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    TokenId,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL, CacheLevel
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
    CachedCudaStream,
    init_cuda_once,
    temporary_sys_path,
)

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from fake_engine import FakeEngine, Role, Step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS_PER_BLOCK = 32
# Bytes per block of one KEY or VALUE buffer. GPU quotas are carved out of CUDA virtual
# memory with a coarse (MiB-level) granularity, so blocks are made large enough that a
# "one block" quota really holds one block.
KV_BUF_SIZE = 1 << 20
BYTES_PER_BLOCK = 2 * KV_BUF_SIZE
HOST_LEVEL = CacheLevel(1)

GPU, HOST, DISK, REMOTE, NONE = range(5)


def _make_config(
    *, gpu_blocks: int, host_blocks: int, sliding_window_size: int | None = None
) -> KVCacheManagerConfig:
    cache_tiers = [GpuCacheTierConfig(quota=gpu_blocks * BYTES_PER_BLOCK)]
    if host_blocks > 0:
        cache_tiers.append(HostCacheTierConfig(quota=host_blocks * BYTES_PER_BLOCK))
    return KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=cache_tiers,
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[
                    BufferConfig(role=Role.KEY, size=KV_BUF_SIZE),
                    BufferConfig(role=Role.VALUE, size=KV_BUF_SIZE),
                ],
                sliding_window_size=sliding_window_size,
                num_sink_tokens=0 if sliding_window_size is not None else None,
            )
        ],
    )


def _segments_by_tier(segments) -> dict[int, int]:
    by_tier: dict[int, int] = {}
    for tier, tokens in segments:
        by_tier[tier] = by_tier.get(tier, 0) + tokens
    return by_tier


class _Harness:
    def __init__(self, cfg: KVCacheManagerConfig) -> None:
        init_cuda_once()
        self.cfg = cfg
        self.engine = FakeEngine(cfg)
        self.manager = KVCacheManager(cfg)
        self._stream_holder = CachedCudaStream()
        self.stream = cast(CudaStream, self._stream_holder.handle)
        self._open_caches = []

    def prefill_and_commit(self, prompt: list) -> None:
        """Run one request over ``prompt`` and leave its blocks in the reuse tree."""
        kv_cache = self.manager.create_kv_cache()
        self._open_caches.append(kv_cache)
        assert kv_cache.resume(self.stream)
        kv_cache.capacity = len(prompt)
        self.engine.execute([Step(kv_cache, prompt, [])], self.stream)
        kv_cache.commit(prompt)
        # Reuse/allocation statistics stay pending on the cache until the request is
        # scheduled (KVCacheManagerV2.commit_scheduled_kv_cache_stats); flush them here.
        kv_cache.commit_pending_stats()
        self.close(kv_cache)
        # Discard the statistics of the priming request.
        self.manager.get_and_reset_iteration_stats()

    def lookup(self, tokens: list):
        """Create a cache from a reuse match (not resumed) and publish its reuse stats."""
        kv_cache = self.manager.create_kv_cache(None, tokens)
        self._open_caches.append(kv_cache)
        kv_cache.commit_pending_stats()
        return kv_cache

    def close(self, kv_cache) -> None:
        if kv_cache in self._open_caches:
            self._open_caches.remove(kv_cache)
            kv_cache.close()

    def reuse_stats(self):
        deltas = self.manager.get_and_reset_iteration_stats()
        totals = {
            "reused": 0,
            "gpu": 0,
            "host": 0,
            "disk": 0,
            "remote": 0,
            "onboard": 0,
        }
        for delta in deltas.values():
            totals["reused"] += delta.iter_reused_blocks
            totals["gpu"] += delta.iter_reused_blocks_gpu
            totals["host"] += delta.iter_reused_blocks_host
            totals["disk"] += delta.iter_reused_blocks_disk
            totals["remote"] += delta.iter_reused_blocks_remote
            totals["onboard"] += delta.iter_onboard_blocks
        return totals

    def shutdown(self) -> None:
        for kv_cache in list(self._open_caches):
            self.close(kv_cache)
        self.manager.shutdown()


@pytest.fixture
def prompt():
    return [TokenId(t) for t in range(4 * TOKENS_PER_BLOCK)]


def test_hits_on_gpu_resident_blocks_are_attributed_to_gpu(prompt):
    harness = _Harness(_make_config(gpu_blocks=16, host_blocks=16))
    try:
        harness.prefill_and_commit(prompt)

        # The second request extends the prompt so the whole committed prefix matches.
        kv_cache = harness.lookup(prompt + [TokenId(10_000)])
        assert kv_cache.num_committed_tokens == len(prompt)
        assert _segments_by_tier(kv_cache.reuse_tier_segments) == {GPU: len(prompt)}

        stats = harness.reuse_stats()
        assert stats["reused"] == 4
        assert stats["gpu"] == 4
        assert stats["host"] == stats["disk"] == stats["remote"] == 0
        committed = harness.manager.get_committed_stats()
        assert committed.reused_blocks_gpu == 4 and committed.reused_blocks_host == 0
        harness.close(kv_cache)
    finally:
        harness.shutdown()


def test_hits_served_from_host_are_attributed_to_host(prompt):
    harness = _Harness(_make_config(gpu_blocks=16, host_blocks=16))
    try:
        harness.prefill_and_commit(prompt)

        # Shrinking the GPU quota below the committed footprint pushes the committed,
        # evictable pages down to the host tier.
        assert harness.manager.resize(GPU_LEVEL, 1 * BYTES_PER_BLOCK)

        kv_cache = harness.lookup(prompt + [TokenId(10_000)])
        assert kv_cache.num_committed_tokens == len(prompt)
        by_tier = _segments_by_tier(kv_cache.reuse_tier_segments)
        # Every matched token has exactly one tier and at least three of the four blocks
        # had to leave the one-block GPU pool.
        assert sum(by_tier.values()) == len(prompt)
        assert set(by_tier) <= {GPU, HOST}
        assert by_tier.get(HOST, 0) >= 3 * TOKENS_PER_BLOCK

        stats = harness.reuse_stats()
        assert stats["reused"] == 4
        assert stats["gpu"] + stats["host"] == 4
        assert stats["host"] >= 3
        assert stats["host"] * TOKENS_PER_BLOCK == by_tier[HOST]
        committed = harness.manager.get_committed_stats()
        assert committed.reused_blocks_host == stats["host"]
        assert committed.reused_blocks_gpu == stats["gpu"]

        # Resuming brings the host pages back: the onboard traffic of this request must
        # match the host attribution decided at match time.
        assert harness.manager.resize(GPU_LEVEL, 16 * BYTES_PER_BLOCK)
        assert kv_cache.resume(harness.stream)
        assert harness.reuse_stats()["onboard"] == stats["host"]
        harness.close(kv_cache)
    finally:
        harness.shutdown()


def test_stale_sliding_window_blocks_are_attributed_to_none(prompt):
    """Sliding-window-only models load only the blocks inside the window.

    The rest of the matched prefix is still skipped by reuse but must be reported as
    'none', not 'gpu'.
    """
    harness = _Harness(
        _make_config(gpu_blocks=16, host_blocks=0, sliding_window_size=TOKENS_PER_BLOCK)
    )
    try:
        harness.prefill_and_commit(prompt)

        kv_cache = harness.lookup(prompt + [TokenId(10_000)])
        matched = kv_cache.num_committed_tokens
        assert matched > 0
        by_tier = _segments_by_tier(kv_cache.reuse_tier_segments)
        assert sum(by_tier.values()) == matched
        assert by_tier.get(NONE, 0) > 0
        assert by_tier.get(GPU, 0) > 0
        assert set(by_tier) == {GPU, NONE}

        stats = harness.reuse_stats()
        # Block counters only cover blocks that were actually loaded.
        assert stats["reused"] == stats["gpu"]
        assert stats["gpu"] * TOKENS_PER_BLOCK == by_tier[GPU]
        harness.close(kv_cache)
    finally:
        harness.shutdown()
