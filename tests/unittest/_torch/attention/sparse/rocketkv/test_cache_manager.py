# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

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
