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
"""Step-ahead KV tier staging (``kv_tier_staging``) against the real core and the fake engine.

The fake engine writes token-derived values into a request's pages and checks them on every
later read, so a request that reads a page before the copy filling it finished, or a copy
that reads a page after someone overwrote it, fails the check. The gated tests hold a request's
stream with a host gate, so the ordering they assert cannot pass just because the GPU was fast.
"""

import gc
import itertools
import random
import unittest
from typing import cast

import test_kv_cache_manager_v2 as kv_test  # type: ignore[import-not-found]
from test_kv_cache_manager_v2 import (  # type: ignore[import-not-found]
    GPU_LEVEL,
    CachedCudaStream,
    CudaStream,
    FakeEngine,
    HostGate,
    KVCacheManager,
    ReuseScope,
    Step,
    TokenId,
    TokenIdExt,
    _KVCache,
    get_cached_cuda_event_type,
    init_cuda_once,
)

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_tier_staging import (
    OffloadHeadroom,
    gpu_backed_reuse_tokens,
    has_lower_tier,
    prefetch_reuse_to_gpu,
)

TOKENS_PER_BLOCK = 32
GATE_ITERS = 60_000


class TestKvTierStaging(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        self._token_ids = itertools.count()
        self._streams: list[CachedCudaStream] = []
        gc.collect()
        gc.disable()

    def tearDown(self) -> None:
        gc.enable()
        for holder in self._streams:
            holder.synchronize()
        if hasattr(self, "manager"):
            try:
                self.manager.shutdown()
            finally:
                del self.manager

    # -- helpers ---------------------------------------------------------------------------

    def _prepare(self, gpu_quota: int = 4 << 20, host_quota: int = 32 << 20) -> None:
        self.cfg = kv_test.create_config(
            TOKENS_PER_BLOCK,
            gpu_quota,
            host_quota,
            0,
            num_layers=2,
            window_size=None,
            sink_tokens=0,
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)
        stats = self.manager.get_storage_statistics(GPU_LEVEL)
        self.assertEqual(len(stats), 1, "one pool group keeps the slot accounting exact")

    def _stream(self) -> CudaStream:
        holder = CachedCudaStream()
        self._streams.append(holder)
        return cast(CudaStream, holder.handle)

    def _tokens(self, n: int) -> list[TokenIdExt]:
        return [TokenId(next(self._token_ids)) for _ in range(n)]

    def _gpu(self):
        return self.manager.get_storage_statistics(GPU_LEVEL)[0]

    def _cached_levels(self, prompt) -> list[int]:
        """Reused tokens of ``prompt`` by the level of their coldest page, without claiming them."""
        probe = self.manager.create_kv_cache(ReuseScope(), prompt)
        try:
            return list(probe.cached_tokens_by_level)
        finally:
            probe.close()

    def _run(self, prompt, stream, close: bool = True) -> _KVCache:
        """Admit ``prompt`` (reusing what is cached), check the reused part, write the rest, commit."""
        kv = self.manager.create_kv_cache(ReuseScope(), prompt)
        self.assertTrue(kv.resume(stream))
        reused = kv.num_committed_tokens
        self.assertTrue(kv.resize(len(prompt)))
        history, new = list(prompt[:reused]), list(prompt[reused:])
        self.engine.execute([Step(kv, new, history)], stream)
        if new:
            kv.commit(new)
        if close:
            kv.close()
        return kv

    def _check(self, prompt, stream) -> None:
        """A fresh request over ``prompt`` reads back exactly what its writer stored."""
        kv = self.manager.create_kv_cache(ReuseScope(), prompt)
        self.assertTrue(kv.resume(stream))
        self.assertEqual(kv.num_committed_tokens, len(prompt))
        self.assertTrue(kv.resize(len(prompt)))
        self.engine.execute([Step(kv, [], list(prompt))], stream)
        kv.close()

    def _churn_until_offloaded(self, prompt, stream, max_rounds: int = 64) -> None:
        """Run unrelated requests until every reused token of ``prompt`` sits below the GPU."""
        for _ in range(max_rounds):
            levels = self._cached_levels(prompt)
            if levels[GPU_LEVEL] == 0 and sum(levels) == len(prompt):
                return
            self._run(self._tokens(4 * TOKENS_PER_BLOCK), stream)
        self.fail(f"prompt was not offloaded: {self._cached_levels(prompt)}")

    def _warm_up(self, stream) -> None:
        """Load every kernel a gated test launches before the gate goes up.

        A first launch can load its module lazily, which synchronizes the context and would
        wait on the gate's spin kernel. Covers the fake engine's fill and check and the
        core's offload and onboard copies; the warm-up prompt then leaves the GPU.
        """
        prompt = self._tokens(2 * TOKENS_PER_BLOCK)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, stream)
        self._check(prompt, stream)
        self._churn_until_offloaded(prompt, stream)
        self.manager.get_storage_statistics(GPU_LEVEL)
        for holder in self._streams:
            holder.synchronize()

    # -- tests -----------------------------------------------------------------------------

    def test_lower_tier_detection(self) -> None:
        self._prepare()
        self.assertTrue(has_lower_tier(self.manager))

    def test_headroom_hands_free_slots_to_the_next_allocation(self) -> None:
        self._prepare()
        stream = self._stream()
        total = self._gpu().total
        prompts = []
        while self._gpu().free >= 4:
            prompts.append(self._tokens(4 * TOKENS_PER_BLOCK))
            self._run(prompts[-1], stream)
        num_blocks = 4
        headroom = OffloadHeadroom(self.manager, num_blocks * TOKENS_PER_BLOCK)
        self.assertTrue(self.manager.is_stats_excluded(headroom._kv_cache_id))

        self.assertTrue(headroom.reserve())
        self.assertTrue(headroom.is_reserved)
        self.assertTrue(headroom.reserve(), "reserving twice holds the same headroom")
        held = self._gpu()
        headroom.release()
        self.assertFalse(headroom.is_reserved)
        released = self._gpu()
        self.assertGreaterEqual(released.free, num_blocks)
        self.assertEqual(released.free, held.free + num_blocks)

        # The next allocation of that size takes the free slots and evicts nothing.
        kv = self.manager.create_kv_cache(ReuseScope(), None)
        self.assertTrue(kv.resume(stream))
        self.assertTrue(kv.resize(num_blocks * TOKENS_PER_BLOCK))
        self.assertEqual(self._gpu().evictable, released.evictable)
        self.engine.execute([Step(kv, self._tokens(num_blocks * TOKENS_PER_BLOCK), [])], stream)
        kv.close()

        # Every earlier prompt, wherever eviction put it, still reads back intact.
        for prompt in prompts:
            self._check(prompt, stream)
        self.assertEqual(self._gpu().total, total)

    def test_headroom_fails_cleanly_when_nothing_can_back_it(self) -> None:
        self._prepare()
        stream = self._stream()
        free = self._gpu().free
        holder = self.manager.create_kv_cache(ReuseScope(), None)
        self.assertTrue(holder.resume(stream))
        self.assertTrue(holder.resize(free * TOKENS_PER_BLOCK))
        headroom = OffloadHeadroom(self.manager, 4 * TOKENS_PER_BLOCK)
        self.assertFalse(headroom.reserve())
        self.assertFalse(headroom.is_reserved)
        self.assertEqual(headroom.num_failed, 1)
        headroom.release()
        holder.close()
        self.assertEqual(self._gpu().free, free)

    def test_a_slot_freed_by_headroom_is_written_only_after_its_offload_read_it(self) -> None:
        """The next owner of a slot the headroom freed writes only after the offload read it.

        The gated writer's pages are the only evictable ones, so the headroom evicts them; the
        slots' next owner must wait for that offload, which waits for the writer.
        """
        self._prepare()
        gated, other, reader = self._stream(), self._stream(), self._stream()
        event_type = get_cached_cuda_event_type()
        self._warm_up(other)
        # Its side stream initializes torch's CUDA state, which synchronizes the device.
        headroom = OffloadHeadroom(self.manager, 8 * TOKENS_PER_BLOCK)
        gate = HostGate()
        try:
            prompt = self._tokens(8 * TOKENS_PER_BLOCK)
            gate.block_stream(gated, GATE_ITERS)
            self._run(prompt, gated)
            # Lock every other slot so the headroom can only take the writer's pages: the pin
            # evicts the older warm-up pages and leaves the writer's, the most recent.
            gpu = self._gpu()
            pin = self.manager.create_kv_cache(ReuseScope(), None)
            self.assertTrue(pin.resume(other))
            self.assertTrue(pin.resize((gpu.free + gpu.evictable - 8) * TOKENS_PER_BLOCK))
            self.assertEqual((self._gpu().free, self._gpu().evictable), (0, 8))
            self.assertEqual(self._cached_levels(prompt)[GPU_LEVEL], len(prompt))

            self.assertTrue(headroom.reserve())
            headroom.release()
            self.assertEqual(self._cached_levels(prompt)[GPU_LEVEL], 0)

            next_owner = self.manager.create_kv_cache(ReuseScope(), None)
            self.assertTrue(next_owner.resume(other))
            self.assertTrue(next_owner.resize(8 * TOKENS_PER_BLOCK))
            self.engine.execute([Step(next_owner, self._tokens(8 * TOKENS_PER_BLOCK), [])], other)
            marker = event_type(other)
            self.assertFalse(marker.query_complete())
            gate.release()
            marker.synchronize()
            next_owner.close()
            pin.close()
            self._check(prompt, reader)
        finally:
            gate.close()

    def test_prefetch_onboards_a_reused_prefix_that_reads_back_intact(self) -> None:
        self._prepare()
        gated, other, reader = self._stream(), self._stream(), self._stream()
        event_type = get_cached_cuda_event_type()
        self._warm_up(other)
        gate = HostGate()
        try:
            prompt = self._tokens(8 * TOKENS_PER_BLOCK)
            gate.block_stream(gated, GATE_ITERS)
            self._run(prompt, gated)
            self._churn_until_offloaded(prompt, other)

            self.assertEqual(gpu_backed_reuse_tokens(self.manager, ReuseScope(), prompt), 0)
            self.assertEqual(prefetch_reuse_to_gpu(self.manager, ReuseScope(), prompt), 0)
            levels = self._cached_levels(prompt)
            self.assertEqual(levels[GPU_LEVEL], len(prompt), levels)
            backed = gpu_backed_reuse_tokens(self.manager, ReuseScope(), prompt)
            self.assertEqual(backed, len(prompt))
            # A second prefetch finds everything on the GPU and issues nothing.
            self.assertEqual(prefetch_reuse_to_gpu(self.manager, ReuseScope(), prompt), len(prompt))

            kv = self.manager.create_kv_cache(ReuseScope(), prompt)
            self.assertTrue(kv.resume(reader))
            self.assertTrue(kv.resize(len(prompt)))
            self.engine.execute([Step(kv, [], list(prompt))], reader)
            marker = event_type(reader)
            self.assertFalse(marker.query_complete())
            gate.release()
            marker.synchronize()
            kv.close()
        finally:
            gate.close()

    def test_prefetch_of_unknown_or_unfittable_prefixes_moves_nothing(self) -> None:
        self._prepare()
        stream = self._stream()
        self.assertEqual(prefetch_reuse_to_gpu(self.manager, ReuseScope(), self._tokens(64)), 0)

        prompt = self._tokens(8 * TOKENS_PER_BLOCK)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, stream)
        pin = self.manager.create_kv_cache(ReuseScope(), None)
        self.assertTrue(pin.resume(stream))
        free_and_evictable = self._gpu().free + self._gpu().evictable
        self.assertTrue(pin.resize(free_and_evictable * TOKENS_PER_BLOCK))
        self.assertIsNone(prefetch_reuse_to_gpu(self.manager, ReuseScope(), prompt))
        self.assertEqual(self._cached_levels(prompt)[GPU_LEVEL], 0)
        pin.close()
        self._check(prompt, stream)

    def test_random_interleaving_keeps_every_conversation_intact(self) -> None:
        """Conversations stay intact under random turns, prefetches and headroom cycles.

        Turns run on several streams while the headroom and prefetch run between steps, as the
        executor runs them; every turn checks its whole history.
        """
        self._prepare()
        rng = random.Random(1234)
        streams = [self._stream() for _ in range(3)]
        headroom = OffloadHeadroom(self.manager, 4 * TOKENS_PER_BLOCK)
        conversations: list[list[TokenIdExt]] = []
        for step in range(48):
            headroom.release()
            stream = rng.choice(streams)
            if conversations and rng.random() < 0.7:
                index = rng.randrange(len(conversations))
                history = conversations[index]
                turn = history + self._tokens(rng.choice([1, 2, 3]) * TOKENS_PER_BLOCK)
                if len(turn) <= 16 * TOKENS_PER_BLOCK:
                    conversations[index] = turn
                self._run(turn, stream)
            else:
                conversations.append(self._tokens(rng.choice([2, 4, 6]) * TOKENS_PER_BLOCK))
                self._run(conversations[-1], stream)
            if conversations and rng.random() < 0.5:
                prefetch_reuse_to_gpu(self.manager, ReuseScope(), rng.choice(conversations))
            headroom.reserve()
        headroom.release()
        self.assertGreater(headroom.num_reserved, 0)
        for history in conversations:
            self._check(history, streams[0])


if __name__ == "__main__":
    unittest.main()
