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
"""Layer-wise host->GPU onboarding: ``resume(defer_onboard=True)`` + ``flush_onboard``.

The fake engine writes token-derived values into a request's pages and checks them on every
later read, so a read that overtakes the copy filling its page fails the check. Gated tests
hold the stream that last wrote the onboarded pages: their offload, and therefore the
onboarding copy, cannot finish until the host releases the gate, so the ordering they assert
cannot pass just because the GPU was fast.
"""

import gc
import itertools
import random
import time
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

TOKENS_PER_BLOCK = 32
NUM_LAYERS = 4
GATE_ITERS = 200_000


class TestKvLayerwiseOnboard(unittest.TestCase):
    layer_groups = NUM_LAYERS

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
            num_layers=NUM_LAYERS,
            window_size=None,
            sink_tokens=0,
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)
        self.manager.set_layerwise_onboard(True, self.layer_groups)

    def _stream(self) -> CudaStream:
        holder = CachedCudaStream()
        self._streams.append(holder)
        return cast(CudaStream, holder.handle)

    def _tokens(self, n: int) -> list[TokenIdExt]:
        return [TokenId(next(self._token_ids)) for _ in range(n)]

    def _cached_levels(self, prompt) -> list[int]:
        probe = self.manager.create_kv_cache(ReuseScope(), prompt)
        try:
            return list(probe.cached_tokens_by_level)
        finally:
            probe.close()

    def _run(self, prompt, stream) -> None:
        """Ordinary (non-deferred) turn: check the reused part, write the rest, commit, close."""
        kv = self.manager.create_kv_cache(ReuseScope(), prompt)
        self.assertTrue(kv.resume(stream))
        reused = kv.num_committed_tokens
        self.assertTrue(kv.resize(len(prompt)))
        history, new = list(prompt[:reused]), list(prompt[reused:])
        self.engine.execute([Step(kv, new, history)], stream)
        if new:
            kv.commit(new)
        kv.close()

    def _admit(self, prompt, stream, num_tokens: int | None = None) -> _KVCache:
        """Deferred resume of ``prompt``; the caller flushes and runs the forward."""
        kv = self.manager.create_kv_cache(ReuseScope(), prompt)
        self.assertTrue(kv.resume(stream, defer_onboard=True))
        self.assertTrue(kv.resize(num_tokens if num_tokens is not None else len(prompt)))
        return kv

    def _check_layer(self, kv: _KVCache, layer_id: int, history, stream) -> None:
        for buf_id in range(len(self.engine.layers[layer_id].buffers)):
            self.engine._check_pages(kv, layer_id, buf_id, 0, history, stream)

    def _forward(self, kv: _KVCache, prompt, stream, commit: bool = True) -> None:
        """Emulate a forward that waits per layer: check each layer after its own event."""
        reused = kv.num_committed_tokens
        history, new = list(prompt[:reused]), list(prompt[reused:])
        for layer_id in sorted(self.engine.layers):
            self.manager.wait_layer_onboard(layer_id, stream)
            self._check_layer(kv, layer_id, history, stream)
        self.manager.wait_all_onboard(stream)
        self.engine.execute([Step(kv, new, history)], stream)
        if commit and new:
            kv.commit(new)

    def _churn_until_offloaded(self, prompt, stream, max_rounds: int = 64) -> None:
        for _ in range(max_rounds):
            levels = self._cached_levels(prompt)
            if levels[GPU_LEVEL] == 0 and sum(levels) == len(prompt):
                return
            self._run(self._tokens(4 * TOKENS_PER_BLOCK), stream)
        self.fail(f"prompt was not offloaded: {self._cached_levels(prompt)}")

    def _check(self, prompt, stream) -> None:
        kv = self.manager.create_kv_cache(ReuseScope(), prompt)
        self.assertTrue(kv.resume(stream))
        self.assertEqual(kv.num_committed_tokens, len(prompt))
        self.assertTrue(kv.resize(len(prompt)))
        self.engine.execute([Step(kv, [], list(prompt))], stream)
        kv.close()

    def _warm_up(self, stream) -> None:
        """Load every kernel the gated tests launch, including the deferred path's."""
        prompt = self._tokens(2 * TOKENS_PER_BLOCK)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, stream)
        kv = self._admit(prompt, stream)
        self.manager.flush_onboard(stream)
        self._forward(kv, prompt, stream)
        kv.close()
        self._churn_until_offloaded(prompt, stream)
        for holder in self._streams:
            holder.synchronize()

    def _wait_complete(self, event, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if event.query_complete():
                return True
            time.sleep(0.001)
        return False

    # -- data ------------------------------------------------------------------------------

    def test_deferred_onboarding_reads_back_intact_per_layer(self) -> None:
        self._prepare()
        stream = self._stream()
        prompt = self._tokens(6 * TOKENS_PER_BLOCK)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, stream)

        turn = prompt + self._tokens(2 * TOKENS_PER_BLOCK)
        kv = self._admit(turn, stream)
        self.assertEqual(kv.num_committed_tokens, len(prompt))
        # Deferred pages already report the GPU level; the data is not there until the flush.
        self.assertEqual(self._cached_levels(prompt)[GPU_LEVEL], len(prompt))
        self.assertTrue(self.manager.has_onboard_events)
        self.assertTrue(self.manager.flush_onboard(stream))
        self.assertFalse(self.manager.flush_onboard(stream), "a second flush has nothing to issue")
        self._forward(kv, turn, stream)
        kv.close()
        self.assertFalse(self.manager.has_onboard_events)
        self._churn_until_offloaded(turn, stream)
        self._check(turn, stream)

    def test_partial_block_reuse_is_copied_from_the_host_source(self) -> None:
        """A reused partial block is copied into a private slot; the writer's stream waits for it."""
        self._prepare()
        stream = self._stream()
        prompt = self._tokens(5 * TOKENS_PER_BLOCK + TOKENS_PER_BLOCK // 2)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, stream)

        turn = prompt + self._tokens(TOKENS_PER_BLOCK)
        kv = self._admit(turn, stream)
        self.assertEqual(kv.num_committed_tokens, len(prompt))
        self.manager.flush_onboard(stream)
        self._forward(kv, turn, stream)
        kv.close()
        self._check(turn, stream)

    def test_shared_prefix_locked_twice_before_the_flush(self) -> None:
        self._prepare()
        stream_a, stream_b = self._stream(), self._stream()
        prompt = self._tokens(4 * TOKENS_PER_BLOCK)
        self._run(prompt, stream_a)
        self._churn_until_offloaded(prompt, stream_a)

        turn_a = prompt + self._tokens(TOKENS_PER_BLOCK)
        turn_b = prompt + self._tokens(TOKENS_PER_BLOCK)
        kv_a = self._admit(turn_a, stream_a)
        kv_b = self._admit(turn_b, stream_b)
        self.assertEqual(kv_b.num_committed_tokens, len(prompt))
        self.manager.flush_onboard(stream_a)
        self.manager.flush_onboard(stream_b)
        # Both forwards wait per layer on the same events.
        for kv, turn, stream in ((kv_a, turn_a, stream_a), (kv_b, turn_b, stream_b)):
            reused = kv.num_committed_tokens
            for layer_id in sorted(self.engine.layers):
                self.manager.wait_layer_onboard(layer_id, stream)
                self._check_layer(kv, layer_id, list(turn[:reused]), stream)
        self.manager.wait_all_onboard(stream_a)
        self.manager.wait_all_onboard(stream_b)
        for kv, turn, stream in ((kv_a, turn_a, stream_a), (kv_b, turn_b, stream_b)):
            reused = kv.num_committed_tokens
            self.engine.execute([Step(kv, list(turn[reused:]), list(turn[:reused]))], stream)
            kv.commit(list(turn[reused:]))
            kv.close()
        self._check(turn_a, stream_a)
        self._check(turn_b, stream_b)

    # -- races -----------------------------------------------------------------------------

    def test_closing_before_the_flush_issues_the_copy_first(self) -> None:
        """A pending page released unread is copied before anything can move or drop it."""
        self._prepare()
        stream = self._stream()
        prompt = self._tokens(6 * TOKENS_PER_BLOCK)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, stream)
        kv = self._admit(prompt + self._tokens(TOKENS_PER_BLOCK), stream)
        kv.close()  # flushes
        self.assertTrue(self.manager.has_onboard_events)
        self.manager.wait_all_onboard(stream)
        # Offload it again (reading the GPU copy) and read it back.
        self._churn_until_offloaded(prompt, stream)
        self._check(prompt, stream)

    def test_suspend_before_the_flush_then_resume(self) -> None:
        self._prepare()
        stream = self._stream()
        prompt = self._tokens(4 * TOKENS_PER_BLOCK)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, stream)
        turn = prompt + self._tokens(TOKENS_PER_BLOCK)
        kv = self._admit(turn, stream)
        kv.suspend()  # flushes
        self._churn_until_offloaded(prompt, stream)
        self.assertTrue(kv.resume(stream, defer_onboard=True))
        self.manager.flush_onboard(stream)
        self._forward(kv, turn, stream)
        kv.close()
        self._check(turn, stream)

    def test_offloads_while_pending_do_not_reuse_the_host_source(self) -> None:
        """Non-deferred traffic between resume and flush evicts to host; held sources survive."""
        self._prepare(gpu_quota=4 << 20, host_quota=8 << 20)
        stream, other = self._stream(), self._stream()
        prompt = self._tokens(6 * TOKENS_PER_BLOCK)
        self._run(prompt, stream)
        self._churn_until_offloaded(prompt, other)
        turn = prompt + self._tokens(TOKENS_PER_BLOCK)
        kv = self._admit(turn, stream)
        # Ordinary turns on another stream, forcing GPU->host offloads and host eviction.
        filler = []
        for _ in range(8):
            filler.append(self._tokens(4 * TOKENS_PER_BLOCK))
            self._run(filler[-1], other)
        self.manager.flush_onboard(stream)
        self._forward(kv, turn, stream)
        kv.close()
        self._check(turn, stream)

    def test_a_decode_resume_of_a_pending_page_flushes_and_waits(self) -> None:
        """A non-deferred lock of a pending page issues the copy and waits for it."""
        self._prepare()
        stream_a, stream_b = self._stream(), self._stream()
        prompt = self._tokens(4 * TOKENS_PER_BLOCK)
        self._run(prompt, stream_a)
        self._churn_until_offloaded(prompt, stream_a)
        kv_a = self._admit(prompt + self._tokens(TOKENS_PER_BLOCK), stream_a)
        kv_b = self.manager.create_kv_cache(ReuseScope(), prompt)
        self.assertTrue(kv_b.resume(stream_b))  # not deferred: flushes and waits
        self.assertTrue(kv_b.resize(len(prompt)))
        self.engine.execute([Step(kv_b, [], list(prompt))], stream_b)
        kv_b.close()
        self.manager.wait_all_onboard(stream_a)
        kv_a.close()

    # -- ordering --------------------------------------------------------------------------

    def test_the_stream_does_not_wait_for_the_migration_but_each_layer_does(self) -> None:
        self._prepare()
        gated, other, reader = self._stream(), self._stream(), self._stream()
        layer_streams = [self._stream() for _ in range(NUM_LAYERS)]
        event_type = get_cached_cuda_event_type()
        self._warm_up(other)
        gate = HostGate()
        try:
            prompt = self._tokens(6 * TOKENS_PER_BLOCK)
            gate.block_stream(gated, GATE_ITERS)
            self._run(prompt, gated)
            self._churn_until_offloaded(prompt, other)

            turn = prompt + self._tokens(TOKENS_PER_BLOCK)
            kv = self._admit(turn, reader)
            self.manager.flush_onboard(reader)
            layer_markers = []
            for layer_id, stream in zip(sorted(self.engine.layers), layer_streams):
                self.manager.wait_layer_onboard(layer_id, stream)
                layer_markers.append(event_type(stream))
            self.assertFalse(any(m.query_complete() for m in layer_markers))
            # The request's own stream is free: it did not wait for the gated migration.
            reader_marker = event_type(reader)
            self.assertTrue(self._wait_complete(reader_marker))
            self.assertFalse(any(m.query_complete() for m in layer_markers))
            gate.release()
            # Markers sit on separate streams, so their completion order says nothing about the
            # copy order (one stream); only that each completes once its layer has landed.
            for marker in layer_markers:
                self.assertTrue(self._wait_complete(marker, timeout_s=10.0))
            self._forward(kv, turn, reader)
            kv.close()
            self._check(turn, other)
        finally:
            gate.close()

    def test_without_deferral_the_stream_waits_for_the_whole_migration(self) -> None:
        self._prepare()
        gated, other, reader = self._stream(), self._stream(), self._stream()
        event_type = get_cached_cuda_event_type()
        self._warm_up(other)
        gate = HostGate()
        try:
            prompt = self._tokens(6 * TOKENS_PER_BLOCK)
            gate.block_stream(gated, GATE_ITERS)
            self._run(prompt, gated)
            self._churn_until_offloaded(prompt, other)
            kv = self.manager.create_kv_cache(ReuseScope(), prompt)
            self.assertTrue(kv.resume(reader))
            marker = event_type(reader)
            time.sleep(0.05)
            self.assertFalse(marker.query_complete())
            gate.release()
            marker.synchronize()
            self.assertTrue(kv.resize(len(prompt)))
            self.engine.execute([Step(kv, [], list(prompt))], reader)
            kv.close()
        finally:
            gate.close()

    # -- soak ------------------------------------------------------------------------------

    def test_random_steps_keep_every_conversation_intact(self) -> None:
        """Steps admit several deferred turns, flush once, then run per-layer forwards."""
        self._prepare()
        rng = random.Random(4321)
        streams = [self._stream() for _ in range(3)]
        conversations: list[list[TokenIdExt]] = []
        for _ in range(40):
            stream = rng.choice(streams)
            admitted = []
            for _ in range(rng.choice([1, 2, 3])):
                if conversations and rng.random() < 0.7:
                    index = rng.randrange(len(conversations))
                    extra = rng.choice(
                        [1, TOKENS_PER_BLOCK // 2, TOKENS_PER_BLOCK, 2 * TOKENS_PER_BLOCK]
                    )
                    turn = conversations[index] + self._tokens(extra)
                    if len(turn) <= 12 * TOKENS_PER_BLOCK:
                        conversations[index] = turn
                else:
                    turn = self._tokens(rng.choice([2, 3, 5]) * TOKENS_PER_BLOCK + rng.randrange(8))
                    conversations.append(turn)
                kv = self.manager.create_kv_cache(ReuseScope(), turn)
                if not kv.resume(stream, defer_onboard=True):
                    kv.close()
                    continue
                if not kv.resize(len(turn)):
                    kv.close()
                    continue
                admitted.append((kv, turn))
            self.manager.flush_onboard(stream)
            for kv, turn in admitted:
                self._forward(kv, turn, stream)
                kv.close()
            if rng.random() < 0.3:
                self._run(self._tokens(4 * TOKENS_PER_BLOCK), rng.choice(streams))
        for history in conversations:
            self._check(history, streams[0])


class TestKvLayerwiseOnboardGrouped(TestKvLayerwiseOnboard):
    """Same contract with several layers per event."""

    layer_groups = 2


if __name__ == "__main__":
    unittest.main()
