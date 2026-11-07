# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytestmark = pytest.mark.cpu_only


def test_kt_pool_uses_local_heads():
    """Match KT allocation and budget for distinct per-layer heads with TP=4."""
    from tensorrt_llm._torch.attention.backends.sparse.rocket import cache_manager as module

    global_heads = [8, 12]
    local_heads = [2, 3]

    # Supply the parent manager's resolved layout without allocating a GPU KV pool.
    def init_parent(self, *args, **kwargs):
        self.num_kv_heads_per_layer = local_heads
        self.num_local_layers = len(local_heads)
        self.blocks_in_primary_pool = 2
        self.tokens_per_block = 64
        self.head_dim = 128
        self.kv_factor = 2
        self.dtype = module.DataType.HALF

    empty = torch.empty

    def cpu_empty(shape, *, device, dtype):
        assert device == "cuda"
        return empty(shape, device="cpu", dtype=dtype)

    params = module.RocketKVParams(kt_cache_dtype="bfloat16", page_size=4)
    config = SimpleNamespace(to_sparse_params=lambda **kwargs: params)
    with (
        patch.object(module.KVCacheManager, "__init__", init_parent),
        patch.object(module, "BlockManager"),
        patch.object(module.torch, "empty", cpu_empty),
    ):
        manager = module.RocketKVCacheManager(
            SimpleNamespace(enable_block_reuse=False),
            None,
            num_layers=len(local_heads),
            num_kv_heads=global_heads,
            head_dim=128,
            tokens_per_block=64,
            max_seq_len=128,
            max_batch_size=1,
            mapping=None,
            sparse_attention_config=config,
        )

    pools = manager.kt_cache_pool_per_layer
    assert [tuple(pool.shape) for pool in pools] == [(2, 16, heads, 256) for heads in local_heads]
    kt_bytes = sum(pool.numel() * pool.element_size() for pool in pools)
    main_bytes_per_token = 2 * sum(local_heads) * 128 * 2
    assert manager.get_cache_bytes_per_token() == main_bytes_per_token + kt_bytes // 128


@pytest.mark.parametrize("is_gen", [False, True])
def test_dummy_requests_without_free_blocks(is_gen: bool) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.rocket import cache_manager as module

    manager = module.RocketKVCacheManager.__new__(module.RocketKVCacheManager)
    manager.kt_cache_manager = SimpleNamespace(add_tokens=Mock())
    with patch.object(module.KVCacheManager, "get_num_free_blocks", return_value=0) as free_blocks:
        requests = manager.add_dummy_requests([1], token_nums=[8], is_gen=is_gen)

    assert requests is None
    free_blocks.assert_called_once_with()
    manager.kt_cache_manager.add_tokens.assert_not_called()


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
def test_registry_follows_kv_cache_manager_version(use_kv_cache_manager_v2: bool) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.registry import (
        get_sparse_attn_kv_cache_manager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.rocket import (
        RocketKVCacheManager,
        RocketKVCacheManagerV2,
    )
    from tensorrt_llm.llmapi import RocketSparseAttentionConfig

    manager_cls = get_sparse_attn_kv_cache_manager(
        RocketSparseAttentionConfig(), use_kv_cache_manager_v2=use_kv_cache_manager_v2
    )

    expected_cls = RocketKVCacheManagerV2 if use_kv_cache_manager_v2 else RocketKVCacheManager
    assert manager_cls is expected_cls


@pytest.mark.parametrize(
    "enable_block_reuse,tokens_per_block,match",
    [
        (True, 32, "block reuse"),
        (False, 30, "multiple of the RocketKV page_size"),
    ],
)
def test_v2_rejects_unsupported_layouts(
    enable_block_reuse: bool, tokens_per_block: int, match: str
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.rocket import cache_manager as module
    from tensorrt_llm.llmapi import RocketSparseAttentionConfig

    with (
        patch.object(module.KVCacheManagerV2, "__init__") as parent_init,
        pytest.raises(ValueError, match=match),
    ):
        module.RocketKVCacheManagerV2(
            SimpleNamespace(enable_block_reuse=enable_block_reuse),
            None,
            num_layers=1,
            num_kv_heads=8,
            head_dim=128,
            tokens_per_block=tokens_per_block,
            max_seq_len=128,
            max_batch_size=1,
            mapping=None,
            sparse_attention_config=RocketSparseAttentionConfig(page_size=4),
        )
    parent_init.assert_not_called()


@pytest.mark.parametrize("kt_cache_dtype,kt_dtype_bytes", [("float8_e5m2", 1), ("bfloat16", 2)])
def test_v2_static_cost_adds_kt_rows(kt_cache_dtype: str, kt_dtype_bytes: int) -> None:
    """The V2 estimate adds the KT bytes the pages hold on top of the plain K/V estimate."""
    from tensorrt_llm._torch.attention.backends.sparse.rocket import RocketKVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi import RocketSparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    num_layers, num_kv_heads, head_dim, tp_size, page_size = 4, 8, 128, 2, 4
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_attention_heads=32,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=32 * head_dim,
        ),
        quant_config=None,
        sparse_attention_config=RocketSparseAttentionConfig(
            page_size=page_size, kt_cache_dtype=kt_cache_dtype
        ),
    )
    kwargs = dict(
        num_layers=num_layers,
        tokens_per_block=32,
        max_seq_len=4096,
        max_batch_size=4,
        max_num_tokens=8192,
    )
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=0)

    dense_slope, dense_intercept = KVCacheManagerV2.get_cache_size_per_token(
        model_config, mapping, **kwargs
    )
    slope, intercept = RocketKVCacheManagerV2.get_cache_size_per_token(
        model_config, mapping, **kwargs
    )

    local_kv_hidden = num_kv_heads // tp_size * head_dim
    assert dense_slope == num_layers * 2 * local_kv_hidden * 2
    kt_bytes_per_token = num_layers * 2 * local_kv_hidden * kt_dtype_bytes // page_size
    assert slope == dense_slope + kt_bytes_per_token
    assert intercept == dense_intercept


def test_v2_keeps_history_ownership_and_rejects_kv_cache_compression() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.rocket import RocketKVCacheManagerV2

    manager = RocketKVCacheManagerV2.__new__(RocketKVCacheManagerV2)
    # The base initializer's default does not take the ownership away.
    manager.kv_compression_manages_history = False
    assert manager.kv_compression_manages_history

    # A KV-cache compression manager binding after construction is rejected.
    manager._constructed = True
    with pytest.raises(NotImplementedError, match="KV-cache compression"):
        manager.kv_compression_manages_history = False
    assert manager.kv_compression_manages_history
