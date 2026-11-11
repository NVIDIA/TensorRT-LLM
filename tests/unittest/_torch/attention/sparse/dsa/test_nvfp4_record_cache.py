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
"""Storage of the ``fp8rope`` NVFP4 DSA KV-cache layout: one inline-scale record per token."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    NVFP4_MLA_KV_CACHE_LAYOUT_ENV,
    DSACacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="NVFP4 KV caches require Blackwell",
)

_NUM_LAYERS = 2
_TOKENS_PER_BLOCK = 64
_RECORD_BYTES = 352
_INDEXER_BYTES = 128 + 4


def _make_manager(sparse_config):
    pretrained_config = SimpleNamespace(
        num_hidden_layers=_NUM_LAYERS, kv_lora_rank=512, qk_rope_head_dim=64
    )
    return DSACacheManagerV2(
        KvCacheConfig(max_tokens=4096, enable_block_reuse=False),
        CacheType.SELFKONLY,
        num_layers=_NUM_LAYERS,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.NVFP4,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )


@pytest.mark.parametrize("layout", ["residual", "fp8rope"])
def test_record_layout_has_no_scale_pool(monkeypatch, layout):
    monkeypatch.setenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, layout)
    sparse_config = DeepSeekSparseAttentionConfig(index_head_dim=128)
    manager = _make_manager(sparse_config)
    try:
        roles = manager._get_layer_roles_by_local_id(0)
        pointers = manager.kv_cache_pool_pointers
        if layout == "residual":
            assert manager.mla_kv_cache_residual_dim == 64
            assert Role.KEY_BLOCK_SCALE in roles
            assert manager.get_layer_bytes_per_token(0, Role.KEY) == 320
            assert manager.get_layer_bytes_per_token(0, Role.KEY_BLOCK_SCALE) == 40
            assert int(pointers[0, 0, 1]) != 0
            latent_bytes = 360
        else:
            assert manager.mla_kv_cache_residual_dim == 128
            assert Role.KEY_BLOCK_SCALE not in roles
            assert manager.get_layer_bytes_per_token(0, Role.KEY) == _RECORD_BYTES
            assert int(pointers[0, 0, 1]) == 0
            latent_bytes = _RECORD_BYTES
            pool = manager.get_unique_primary_pool()
            assert pool.numel() == pool.shape[0] * _NUM_LAYERS * _TOKENS_PER_BLOCK * _RECORD_BYTES
        assert manager.get_layer_bytes_per_token(0, Role.ALL) == latent_bytes + _INDEXER_BYTES
        assert manager.get_cache_bytes_per_token() == _NUM_LAYERS * (latent_bytes + _INDEXER_BYTES)

        model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(
                num_hidden_layers=_NUM_LAYERS, kv_lora_rank=512, qk_rope_head_dim=64
            ),
            sparse_attention_config=sparse_config,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
        )
        estimate = DSACacheManagerV2.get_cache_size_per_token(
            model_config, Mapping(), num_layers=_NUM_LAYERS
        )
        assert estimate == manager.get_cache_bytes_per_token()
    finally:
        manager.shutdown()


def test_record_layout_rejects_kv_cache_offload(monkeypatch):
    """The offload arena reads NVFP4 latents with a block-scale pool, which records lack."""
    monkeypatch.setenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, "fp8rope")
    sparse_config = DeepSeekSparseAttentionConfig(index_head_dim=128, enable_kv_cache_offload=True)
    with pytest.raises(ValueError, match="KV cache offload"):
        _make_manager(sparse_config)
