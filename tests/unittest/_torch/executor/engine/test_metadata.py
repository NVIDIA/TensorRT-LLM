# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.model_config import KVCacheLayerSpec, ModelConfig
from tensorrt_llm._torch.pyexecutor.engine.metadata import (
    _get_max_num_heads_per_kv,
    build_attention_metadata,
)

pytestmark = pytest.mark.cpu_only


class _AttentionMetadata:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _AttentionBackend:
    Metadata = _AttentionMetadata


def test_build_attention_metadata_resolves_model_derived_values() -> None:
    sparse_metadata_params = object()
    sparse_attention_config = SimpleNamespace(
        to_sparse_metadata_params=Mock(return_value=sparse_metadata_params)
    )
    pretrained_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=[4, 2],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    model_config = SimpleNamespace(
        pretrained_config=pretrained_config,
        sparse_attention_config=sparse_attention_config,
        enable_flash_mla=True,
    )
    runtime_features = AttentionRuntimeFeatures(cache_reuse=True)

    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=_AttentionBackend,
        attention_runtime_features=runtime_features,
        mapping=object(),
        cache_indirection=None,
    )

    assert metadata.max_num_heads_per_kv == 4
    assert metadata.enable_flash_mla
    assert metadata.enable_context_mla_with_cached_kv
    assert metadata.sparse_metadata_params is sparse_metadata_params
    sparse_attention_config.to_sparse_metadata_params.assert_called_once_with(
        pretrained_config=pretrained_config
    )


def test_build_attention_metadata_forwards_shared_and_cache_inputs() -> None:
    mapping = object()
    cache_indirection = object()
    kv_cache_manager = object()
    draft_kv_cache_manager = object()
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        sparse_attention_config=None,
        enable_flash_mla=True,
    )

    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(),
        mapping=mapping,
        cache_indirection=cache_indirection,
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=draft_kv_cache_manager,
        enable_context_mla_with_cached_kv=True,
        max_num_heads_per_kv=4,
    )

    assert metadata.max_num_requests == 4
    assert metadata.max_num_tokens == 16
    assert metadata.max_num_sequences == 8
    assert metadata.mapping is mapping
    assert metadata.cache_indirection is cache_indirection
    assert metadata.kv_cache_manager is kv_cache_manager
    assert metadata.draft_kv_cache_manager is draft_kv_cache_manager
    assert metadata.enable_context_mla_with_cached_kv
    assert metadata.max_num_heads_per_kv == 4


@pytest.mark.parametrize(
    ("head_pairs", "expected_ratio"),
    [([(16, 8), (16, 1)], 16), ([(16, 8), (8, 2)], 4)],
)
def test_metadata_gqa_ratio_uses_model_provided_per_layer_heads(
    head_pairs: list[tuple[int, int]], expected_ratio: int
) -> None:
    class StrictConfig:
        num_hidden_layers = 2

        def __getattribute__(self, name: str) -> object:
            if name in {"num_attention_heads", "num_key_value_heads"}:
                raise RuntimeError(f"global geometry must not be read: {name}")
            return super().__getattribute__(name)

    model_config = ModelConfig(pretrained_config=StrictConfig())
    model_config.set_kv_cache_layer_specs(
        [
            KVCacheLayerSpec(head_dim=256, num_q_heads=q_heads, num_kv_heads=kv_heads)
            for q_heads, kv_heads in head_pairs
        ]
    )
    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=1,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(),
        mapping=object(),
        cache_indirection=None,
    )

    assert metadata.max_num_heads_per_kv == expected_ratio


@pytest.mark.parametrize(
    ("config", "expected_ratio"),
    [
        (SimpleNamespace(num_attention_heads=8, num_key_value_heads=2), 4),
        (SimpleNamespace(num_attention_heads=8, num_key_value_heads=[4, 2]), 4),
        (SimpleNamespace(num_attention_heads=8, num_key_value_heads=[None, 0]), 1),
        (SimpleNamespace(), 1),
    ],
)
def test_metadata_gqa_ratio_preserves_flat_config_fallback(
    config: SimpleNamespace, expected_ratio: int
) -> None:
    assert _get_max_num_heads_per_kv(SimpleNamespace(pretrained_config=config)) == expected_ratio
