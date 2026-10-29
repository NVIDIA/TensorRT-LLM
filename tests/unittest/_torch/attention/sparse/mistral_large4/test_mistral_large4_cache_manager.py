# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral-Large-4 latent cache's sizing and selection, from the checkpoint's own layer
structure. No device: the per-token bytes the engine budgets with and the registry branch."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse import get_sparse_attn_kv_cache_manager
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4.cache_manager import (
    TOKEN_BYTES,
    kv_source_layers,
)
from tensorrt_llm._torch.configs.mistral_large4 import MistralLarge4Config
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

pytestmark = pytest.mark.cpu_only

# The Mistral-Large-4-1T-A52B checkpoints' layer structure (params.json): 72 layers; every
# fourth from 39 on attends over layer 35's latent; the indexer is 'full' on layers 0-16 and
# then on two of every four.
_SOURCES = [i for i in range(72)]
for _i in range(39, 72, 4):
    _SOURCES[_i] = 35
_SHARING = ["full"] * 17 + [
    ("shared", "shared", "full", "full")[(i - 17) % 4] for i in range(17, 72)
]


def _model_config(sources=_SOURCES, sharing=_SHARING):
    cfg = MistralLarge4Config(
        num_hidden_layers=len(sources),
        kv_source_layers=sources,
        index_sharing=sharing,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    sparse = DeepSeekSparseAttentionConfig(index_n_heads=32, index_head_dim=128, index_topk=2048)
    return SimpleNamespace(pretrained_config=cfg, sparse_attention_config=sparse)


def test_bytes_per_token_are_owners_times_656_plus_full_indexers_times_132():
    """63 latent owners and 44 full-indexer layers: 63 x 656 + 44 x 132 = 47,136 B/token,
    against 72 x 576 + 44 x 132 = 47,280 for the 576-byte pool on every layer."""
    mc = _model_config()
    owners = sum(1 for i, s in enumerate(_SOURCES) if s == i)
    full = _SHARING.count("full")
    assert (owners, full, TOKEN_BYTES) == (63, 44, 656)
    mapping = SimpleNamespace(pp_size=1)
    assert MistralLarge4CacheManager.get_cache_size_per_token(mc, mapping) == 63 * 656 + 44 * 132


@pytest.mark.parametrize(
    "sources, match",
    [
        ([0, 2, 2], "earlier"),  # a source later than its reader
        ([0, 0, 1], "own its latent"),  # a source that is itself a sharer
        ([0, 1], "entries"),  # a length other than num_hidden_layers
    ],
)
def test_kv_source_layers_are_validated(sources, match):
    cfg = SimpleNamespace(kv_source_layers=sources, num_hidden_layers=3)
    with pytest.raises(ValueError, match=match):
        kv_source_layers(cfg)


def test_registry_selects_the_manager_from_the_config_class():
    """The DSA branch hands a MistralLarge4Config checkpoint to this manager (V2 only) and
    every other DSA checkpoint to the generic one, whatever attributes it carries."""
    mc = _model_config()
    sparse, cfg = mc.sparse_attention_config, mc.pretrained_config
    assert get_sparse_attn_kv_cache_manager(sparse, True, cfg) is MistralLarge4CacheManager
    assert get_sparse_attn_kv_cache_manager(sparse, True, SimpleNamespace()) is DSACacheManagerV2
    other = SimpleNamespace(kv_source_layers=_SOURCES)
    assert get_sparse_attn_kv_cache_manager(sparse, True, other) is DSACacheManagerV2
    assert get_sparse_attn_kv_cache_manager(sparse, True) is DSACacheManagerV2
    with pytest.raises(ValueError, match="V2"):
        get_sparse_attn_kv_cache_manager(sparse, False, cfg)
