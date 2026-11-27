# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral-Large-4 latent cache's sizing and selection, from the checkpoint's own layer
structure. No device: the per-token bytes the engine budgets with and the class it builds."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse import get_sparse_attn_kv_cache_manager
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager, DSACacheManagerV2
from tensorrt_llm._torch.configs.mistral_large4 import MistralLarge4Config
from tensorrt_llm._torch.modeling_v2.models import mistral_large4
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.cache.cache_manager import (
    TOKEN_BYTES,
    MistralLarge4CacheManager,
)
from tensorrt_llm._torch.pyexecutor._util import model_declared_kv_cache_manager_cls
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


_TARGETS = sorted(
    p.name for p in Path(mistral_large4.__file__).parent.glob("mistral_large_4_*") if p.is_dir()
)


@pytest.mark.parametrize("target", _TARGETS)
def test_every_target_declares_the_manager_over_the_dsa_v2_one(target):
    """The engine resolves the DSA manager for the checkpoint's sparse config, and builds the
    subclass a target declares (RuntimeHooks.kv_cache_manager_cls) in its place. Every target
    declares this manager, a DSACacheManagerV2; without KV cache manager V2 the engine resolves
    the V1 DSA manager and refuses it."""
    module = importlib.import_module(f"{mistral_large4.__name__}.{target}.modeling")
    (target_cls,) = [
        cls
        for name, cls in vars(module).items()
        if name.startswith("ModelingV2MistralLarge4") and isinstance(cls, type)
    ]
    sparse = _model_config().sparse_attention_config
    resolved = get_sparse_attn_kv_cache_manager(sparse, True)
    assert resolved is DSACacheManagerV2
    assert model_declared_kv_cache_manager_cls(target_cls, resolved) is MistralLarge4CacheManager
    resolved = get_sparse_attn_kv_cache_manager(sparse, False)
    assert resolved is DSACacheManager
    with pytest.raises(ValueError, match="not a subclass of DSACacheManager"):
        model_declared_kv_cache_manager_cls(target_cls, resolved)
