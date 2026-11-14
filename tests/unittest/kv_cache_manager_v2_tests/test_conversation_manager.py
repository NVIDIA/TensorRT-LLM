# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversation completion ordering and native block-key retention checks."""

from collections import deque
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import ConversationManager
from tensorrt_llm.runtime import kv_cache_manager_v2 as native


def _request(request_id: int, salt: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        is_dummy_request=False,
        py_conversation_params=SimpleNamespace(conversation_id="shared"),
        cache_salt=salt,
        lora_task_id=None,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("completion", ["finish", "empty_plan", "clear"])
def test_overlapping_completion_after_owner_cleanup(completion: str) -> None:
    manager = ConversationManager(max_num_turns=1)
    owner, overlapping = _request(1), _request(2)
    manager.prepare_request(owner)
    manager.prepare_request(overlapping)
    cache = MagicMock(num_committed_tokens=4, tokens_per_block=4)
    cache.plan_committed_block_drop.return_value = None
    if completion == "finish":
        manager.finish_request(owner)
    elif completion == "empty_plan":
        manager.save_drop_plan(owner, cache)
    else:
        manager.clear()
    cache.reset_mock()
    manager.save_drop_plan(overlapping, cache)
    cache.plan_committed_block_drop.assert_not_called()
    assert not manager._conversation_states


@pytest.fixture
def native_cache() -> Iterator[SimpleNamespace]:
    torch.empty(1, device="cuda")
    stream = torch.cuda.Stream()
    manager = native.KVCacheManager(
        native.KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[native.GpuCacheTierConfig(quota=1 << 20)],
            layers=[
                native.AttentionLayerConfig(
                    layer_id=0,
                    buffers=[native.BufferConfig(role="key", size=256)],
                    sliding_window_size=8,
                )
            ],
        )
    )
    conversation = ConversationManager(max_num_turns=1)
    caches = []

    def create(tokens: list[int], salt: str | None = None) -> native._KVCache:
        cache = manager.create_kv_cache(native.ReuseScope(salt=salt), tokens)
        caches.append(cache)
        assert cache.resume(stream.cuda_stream)
        assert cache.resize(len(tokens), len(tokens))
        cache.commit(tokens[cache.num_committed_tokens :])
        cache.stop_committing()
        return cache

    try:
        yield SimpleNamespace(manager=manager, conversation=conversation, create=create)
    finally:
        conversation.clear()
        for cache in caches:
            if cache.status != native._KVCache.Status.CLOSED:
                cache.close()
        manager.shutdown()


def test_cached_block_keys_match_native_hash_chain(native_cache: SimpleNamespace) -> None:
    tokens = list(range(16))
    cache = native_cache.create(tokens, "scope")
    for length in (4, 8, 16):
        keys = native.sequence_to_blockchain_keys(4, cache.reuse_scope, tokens[:length])
        assert cache._get_committed_prefix_key(length) == deque(keys, maxlen=1)[0][1]
    assert cache._get_committed_prefix_key(0) is None
    assert cache._get_committed_prefix_key(20) is None
    cache.close()
    assert cache._get_committed_prefix_key(4) is None


@pytest.mark.parametrize(
    "candidate,salt,advances",
    [
        (list(range(100, 116)), "scope", False),
        (list(range(4)) + list(range(100, 112)), "scope", False),
        (list(range(4)), "scope", False),
        (list(range(16)) + list(range(100, 108)), "scope", True),
        (list(range(16)), "other-scope", False),
    ],
)
def test_complete_block_prefix_controls_retention(
    native_cache: SimpleNamespace, candidate: list[int], salt: str, advances: bool
) -> None:
    manager, conversation = native_cache.manager, native_cache.conversation
    tokens = list(range(16))
    first = native_cache.create(tokens, "scope")
    conversation.prepare_request(_request(1, "scope"))
    conversation.save_drop_plan(_request(1, "scope"), first)
    first.close()
    scope = native.ReuseScope(salt="scope")
    assert manager.probe_reuse(scope, tokens) == len(tokens)

    second = native_cache.create(candidate, salt)
    conversation.prepare_request(_request(2, salt))
    conversation.save_drop_plan(_request(2, salt), second)
    second.close()
    assert manager.probe_reuse(scope, tokens) == (0 if advances else len(tokens))
