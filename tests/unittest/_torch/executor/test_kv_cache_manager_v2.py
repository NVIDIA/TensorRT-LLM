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

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import GpuCacheTierConfig, KVCacheManagerConfig


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int) -> None:
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens: list[int] | None = None
        self.stopped_committing = False

    def commit(self, tokens: list[int]) -> None:
        self.committed_tokens = tokens
        self.num_committed_tokens += len(tokens)

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
        is_dummy_request=False,
        context_current_position=10,
        context_remaining_length=0,
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
    assert kv_cache.stopped_committing


def test_disagg_role_mapper_kinds_default_to_indexed():
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role

    manager = object.__new__(KVCacheManagerV2)

    # K/V default to the TRTLLM head-major layout; the index-key side cache
    # defaults to REPLICATED (every shipped index-K — DSA V1, MiniMax M3 —
    # is TP-replicated). The INDEX_KEY entry is inert unless a subclass
    # registers such buffers.
    assert manager.get_disagg_role_mapper_kinds() == {
        Role.ALL: MapperKind.INDEXED,
        Role.INDEX_KEY: MapperKind.REPLICATED,
    }
