# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-page isolation when conversation IDs collide across overlapping requests."""

import ctypes
import unittest
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import ConversationManager
from tensorrt_llm.runtime import kv_cache_manager_v2 as native

runtime = ctypes.CDLL("libcudart.so")
runtime.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
QUOTA = 256 << 10
PAGE_BYTES = 4096


class TestConversationKVIsolation(unittest.TestCase):
    def setUp(self) -> None:
        torch.empty(1, device="cuda")
        self.stream = torch.cuda.Stream()
        self.manager = native.KVCacheManager(
            native.KVCacheManagerConfig(
                tokens_per_block=4,
                cache_tiers=[native.GpuCacheTierConfig(quota=QUOTA)],
                layers=[
                    native.AttentionLayerConfig(
                        layer_id=layer,
                        buffers=[native.BufferConfig(role="key", size=PAGE_BYTES)],
                        sliding_window_size=None if layer == 0 else 8,
                    )
                    for layer in (0, 1)
                ],
                enable_partial_reuse=False,
            )
        )
        self.conversation = ConversationManager(max_num_turns=1)
        self.caches: list[native._KVCache] = []
        self.next_id = 0

    def tearDown(self) -> None:
        self.conversation.clear()
        for cache in self.caches:
            if cache.status != native._KVCache.Status.CLOSED:
                cache.close()
        torch.cuda.synchronize()
        self.manager.shutdown()

    def addresses(self, cache: native._KVCache) -> dict[tuple[int, int], int]:
        result = {}
        for layer in (0, 1):
            group = self.manager.get_layer_group_id(layer)
            base = self.manager.get_mem_pool_base_address(layer, "key")
            stride = self.manager.get_page_stride(layer, "key")
            for ordinal, index in enumerate(cache.get_base_page_indices(group)):
                index = int(index)
                if index >= 0:
                    result[layer, ordinal] = base + index * stride
        return result

    def paint(self, cache: native._KVCache, value: int, exclude: set[int] | None = None) -> None:
        for address in self.addresses(cache).values():
            if exclude is None or address not in exclude:
                self.assertEqual(runtime.cudaMemset(address, value, PAGE_BYTES), 0)
        torch.cuda.synchronize()

    def snapshot(self, cache: native._KVCache) -> dict[tuple[int, int], bytes]:
        torch.cuda.synchronize()
        result = {}
        for key, address in self.addresses(cache).items():
            data = (ctypes.c_ubyte * PAGE_BYTES)()
            self.assertEqual(runtime.cudaMemcpy(data, address, PAGE_BYTES, 2), 0)
            result[key] = bytes(data)
        self.assertTrue(result, "No actual GPU pages inspected")
        return result

    def create(
        self, tokens: list[int], salt: str | None = None
    ) -> tuple[SimpleNamespace, native._KVCache]:
        self.next_id += 1
        request = SimpleNamespace(
            py_request_id=self.next_id,
            is_dummy_request=False,
            py_conversation_params=SimpleNamespace(conversation_id="DYNAMO_SESSION_ID"),
            cache_salt=salt,
            lora_task_id=None,
        )
        cache = self.manager.create_kv_cache(native.ReuseScope(salt=salt), tokens, id=self.next_id)
        self.caches.append(cache)
        self.assertTrue(cache.resume(self.stream.cuda_stream))
        self.assertTrue(cache.resize(len(tokens), len(tokens)))
        return request, cache

    def commit(self, cache: native._KVCache, tokens: list[int]) -> None:
        remaining = tokens[cache.num_committed_tokens :]
        if remaining:
            cache.commit(remaining)
        cache.stop_committing()

    def pressure(self) -> None:
        for index in range(100):
            tokens = list(range(10_000 + 32 * index, 10_016 + 32 * index))
            _, cache = self.create(tokens, "pressure")
            self.paint(cache, 0x77)
            self.commit(cache, tokens)
            cache.close()

    def test_same_session_different_prompts_never_alias(self) -> None:
        a_tokens, b_tokens = list(range(16)), list(range(100, 116))
        a_req, a = self.create(a_tokens)
        b_req, b = self.create(b_tokens)
        self.conversation.prepare_request(a_req)
        self.conversation.prepare_request(b_req)
        self.assertEqual(b.num_committed_tokens, 0)
        self.assertTrue(set(self.addresses(a).values()).isdisjoint(self.addresses(b).values()))
        self.paint(a, 0x11)
        self.paint(b, 0x22)
        self.commit(a, a_tokens)
        self.commit(b, b_tokens)
        before = self.snapshot(b)
        self.conversation.save_drop_plan(a_req, a)
        a.close()
        self.conversation.save_drop_plan(b_req, b)
        self.pressure()
        self.assertEqual(before, self.snapshot(b))
        self.assertTrue(all(data == bytes([0x22]) * PAGE_BYTES for data in before.values()))

    def test_distinct_tenant_salts_do_not_reuse_identical_tokens(self) -> None:
        tokens = list(range(16))
        a_req, a = self.create(tokens, "tenant-A")
        self.conversation.prepare_request(a_req)
        self.paint(a, 0x11)
        self.commit(a, tokens)
        before = self.snapshot(a)
        self.conversation.save_drop_plan(a_req, a)
        self.assertEqual(self.manager.probe_reuse(native.ReuseScope(salt="tenant-B"), tokens), 0)
        b_req, b = self.create(tokens, "tenant-B")
        self.assertEqual(b.num_committed_tokens, 0)
        self.conversation.prepare_request(b_req)
        self.paint(b, 0x22)
        self.commit(b, tokens)
        self.conversation.save_drop_plan(b_req, b)
        self.assertEqual(before, self.snapshot(a))
        self.assertTrue(set(self.addresses(a).values()).isdisjoint(self.addresses(b).values()))

    def test_dropping_prior_plans_preserves_active_shared_pages_under_pressure(self) -> None:
        tokens = list(range(16))
        a_req, a = self.create(tokens)
        self.conversation.prepare_request(a_req)
        self.paint(a, 0x33)
        self.commit(a, tokens)
        self.conversation.save_drop_plan(a_req, a)
        a.close()
        b_req, b = self.create(tokens)
        self.assertGreater(b.num_committed_tokens, 0, "Shared-prefix path was not exercised")
        self.conversation.prepare_request(b_req)
        self.commit(b, tokens)
        before = self.snapshot(b)
        self.conversation.save_drop_plan(b_req, b)
        c_tokens = tokens + list(range(200, 208))
        c_req, c = self.create(c_tokens)
        self.conversation.prepare_request(c_req)
        self.paint(c, 0x44, set(self.addresses(b).values()))
        self.commit(c, c_tokens)
        self.conversation.save_drop_plan(c_req, c)
        c.close()
        self.pressure()
        self.assertEqual(before, self.snapshot(b))
        self.assertTrue(all(data == bytes([0x33]) * PAGE_BYTES for data in before.values()))

    def test_owner_cancellation_preserves_overlapping_cache(self) -> None:
        tokens = list(range(16))
        a_req, a = self.create(tokens)
        b_req, b = self.create(list(range(100, 116)))
        self.conversation.prepare_request(a_req)
        self.conversation.prepare_request(b_req)
        self.paint(a, 0x11)
        self.paint(b, 0x55)
        self.commit(b, list(range(100, 116)))
        before = self.snapshot(b)
        self.conversation.finish_request(a_req)
        a.close()
        self.conversation.save_drop_plan(b_req, b)
        self.pressure()
        self.assertEqual(before, self.snapshot(b))

    def test_session_collision_preserves_another_tenant_retained_prefix(self) -> None:
        tokens = list(range(16))
        a_req, a = self.create(tokens, "tenant-A")
        self.conversation.prepare_request(a_req)
        self.paint(a, 0x11)
        self.commit(a, tokens)
        self.conversation.save_drop_plan(a_req, a)
        a.close()
        scope = native.ReuseScope(salt="tenant-A")
        before = self.manager.probe_reuse(scope, tokens)
        self.assertGreater(before, 0)
        b_tokens = list(range(100, 116))
        b_req, b = self.create(b_tokens, "tenant-B")
        self.conversation.prepare_request(b_req)
        self.paint(b, 0x22)
        self.commit(b, b_tokens)
        self.conversation.save_drop_plan(b_req, b)
        after = self.manager.probe_reuse(scope, tokens)
        self.assertEqual(after, before, "Session collision evicted another namespace's prefix")

    def test_unrelated_history_preserves_retention_with_same_session_and_scope(self) -> None:
        tokens = list(range(16))
        a_req, a = self.create(tokens)
        self.conversation.prepare_request(a_req)
        self.paint(a, 0x11)
        self.commit(a, tokens)
        a_addresses = set(self.addresses(a).values())
        self.conversation.save_drop_plan(a_req, a)
        a.close()
        before = self.manager.probe_reuse(None, tokens)
        self.assertEqual(before, len(tokens))

        for b_tokens in (
            list(range(100, 116)),
            tokens[:8] + list(range(100, 108)),
            tokens[:8],
        ):
            with self.subTest(tokens=b_tokens):
                b_req, b = self.create(b_tokens)
                self.conversation.prepare_request(b_req)
                self.paint(b, 0x22, a_addresses)
                self.commit(b, b_tokens)
                self.conversation.save_drop_plan(b_req, b)
                b.close()
                self.assertEqual(self.manager.probe_reuse(None, tokens), before)

    def test_continuation_advances_retention_after_full_prefix_verification(self) -> None:
        tokens = list(range(16))
        a_req, a = self.create(tokens)
        self.conversation.prepare_request(a_req)
        self.paint(a, 0x11)
        self.commit(a, tokens)
        a_addresses = set(self.addresses(a).values())
        self.conversation.save_drop_plan(a_req, a)
        a.close()

        continuation = tokens + list(range(100, 108))
        b_req, b = self.create(continuation)
        self.conversation.prepare_request(b_req)
        self.paint(b, 0x22, a_addresses)
        self.commit(b, continuation)
        self.conversation.save_drop_plan(b_req, b)
        b.close()

        self.assertEqual(self.manager.probe_reuse(None, tokens), 0)
        self.assertEqual(self.manager.probe_reuse(None, continuation), len(continuation))
