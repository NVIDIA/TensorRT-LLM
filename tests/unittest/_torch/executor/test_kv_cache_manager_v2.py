# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import GpuCacheTierConfig, KVCacheManagerConfig


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int) -> None:
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens: list[int] | None = None
        self.history_length = 0
        self.capacity = num_committed_tokens
        self.stopped_committing = False

    def commit(self, tokens: list[int]) -> None:
        self.committed_tokens = tokens
        self.num_committed_tokens += len(tokens)

    def resize(self, capacity: int, history_length: int | None = None) -> bool:
        self.capacity = capacity
        self.history_length = history_length
        return True

    def stop_committing(self) -> None:
        self.stopped_committing = True


def _build_cache_config_for_test(
    kv_cache_config: KvCacheConfig, *, is_draft: bool = False
) -> KVCacheManagerConfig:
    cache_manager = object.__new__(KVCacheManagerV2)
    cache_manager.kv_cache_type = CacheTypeCpp.SELFKONLY
    cache_manager.head_dim_per_layer = [128]
    cache_manager.enable_swa_scratch_reuse = False
    cache_manager.num_extra_kv_tokens = 0
    cache_manager.enable_stats = False
    cache_manager.block_reuse_policy = BlockReusePolicy(kv_cache_config.block_reuse_policy)
    cache_manager.is_draft = is_draft
    cache_manager.num_local_layers = 1
    cache_manager.pp_layers = [0]
    cache_manager.max_attention_window_vec = [None]
    cache_manager.get_layer_bytes_per_token = lambda **_: 128

    return cache_manager._build_cache_config(
        kv_cache_config,
        tokens_per_block=128,
        vocab_size=129280,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
    )


@pytest.mark.parametrize(
    ("enable_block_reuse", "block_reuse_policy", "is_draft", "commit_min_snapshot"),
    [
        (True, "all_reusable", False, False),
        (True, "per_request", False, True),
        (False, "per_request", False, False),
        (True, "per_request", True, True),
    ],
)
def test_commit_min_snapshot_follows_block_reuse_policy(
    enable_block_reuse: bool,
    block_reuse_policy: str,
    is_draft: bool,
    commit_min_snapshot: bool,
) -> None:
    config = _build_cache_config_for_test(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            block_reuse_policy=block_reuse_policy,
            enable_partial_reuse=True,
        ),
        is_draft=is_draft,
    )

    assert config.commit_min_snapshot is commit_min_snapshot
    assert config.enable_partial_reuse


@pytest.mark.parametrize("enable_partial_reuse", [False, True])
def test_propagates_partial_reuse_config(enable_partial_reuse: bool) -> None:
    config = _build_cache_config_for_test(KvCacheConfig(enable_partial_reuse=enable_partial_reuse))

    assert config.enable_partial_reuse is enable_partial_reuse


def test_try_commit_blocks_commits_partial_block_at_context_end() -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy=False,
        is_dummy_request=False,
        context_current_position=10,
        context_remaining_length=0,
        block_reuse_commit_limit=lambda: 10,
        get_tokens=lambda beam_id: list(range(10)),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert kv_cache.committed_tokens == [4, 5, 6, 7, 8, 9]
    assert kv_cache.num_committed_tokens == 10
    assert kv_cache.history_length == 10
    assert kv_cache.stopped_committing


def test_try_commit_blocks_stops_at_reusable_prompt_boundary() -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy=False,
        is_dummy_request=False,
        context_current_position=10,
        context_remaining_length=0,
        block_reuse_commit_limit=lambda: 8,
        get_tokens=lambda beam_id: list(range(10)),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert kv_cache.committed_tokens == [4, 5, 6, 7]
    assert kv_cache.num_committed_tokens == 8
    assert kv_cache.history_length == 10
    assert kv_cache.stopped_committing


@pytest.mark.parametrize(("position", "should_commit"), [(32, False), (64, True)])
def test_update_context_resources_commits_only_at_snapshot_boundary(
    position: int, should_commit: bool
) -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        context_current_position=position,
        context_remaining_length=128 - position,
        should_save_ssm_snapshot=lambda commit_end: commit_end == 64,
        next_expected_snapshot_point=lambda: 64 if position < 64 else 128,
    )
    kv_cache = SimpleNamespace(
        is_active=True,
        resize=MagicMock(return_value=True),
        enable_swa_scratch_reuse=True,
    )
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.block_reuse_policy = BlockReusePolicy.PER_REQUEST
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager.try_commit_blocks = MagicMock()

    manager.update_context_resources(SimpleNamespace(context_requests=[request]))

    kv_cache.resize.assert_not_called()
    if should_commit:
        manager.try_commit_blocks.assert_called_once_with(request)
    else:
        manager.try_commit_blocks.assert_not_called()
