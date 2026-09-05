# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention.backends.sparse.hooks import get_sparse_attention_hooks
from tensorrt_llm._torch.attention.backends.sparse.qsa import (
    QSAAttentionMetadata,
    QSAMambaHybridCacheManagerV2,
    QSASparseMetadataParams,
    QSASparseParams,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.module import QSASparseHooks
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import MambaKVCacheParams
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, QSASparseAttentionConfig


def _sparse_params() -> QSASparseParams:
    return QSASparseParams(
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        token_topk=2048,
        compress_ratio=4,
    )


def test_qsa_config_is_exported_from_llmapi() -> None:
    from tensorrt_llm.llmapi import QSASparseAttentionConfig as PublicQSAConfig

    assert PublicQSAConfig is QSASparseAttentionConfig


def test_qsa_config_uses_checkpoint_geometry_without_mutating_user_config() -> None:
    checkpoint_config = SimpleNamespace(
        indexer_n_heads=6,
        indexer_kv_heads=1,
        indexer_head_dim=96,
        indexer_budget=1024,
        indexer_compress_ratio=8,
    )

    sparse_config = QSASparseAttentionConfig()
    initial_config = sparse_config.model_dump()
    params = sparse_config.to_sparse_params(pretrained_config=checkpoint_config)

    assert params == QSASparseParams(
        index_n_heads=6,
        index_kv_heads=1,
        index_head_dim=96,
        token_topk=1024,
        compress_ratio=8,
        seq_len_threshold=1024,
    )
    assert sparse_config.seq_len_threshold is None
    assert sparse_config.model_dump() == initial_config


def test_qsa_config_rejects_missing_checkpoint_geometry() -> None:
    with pytest.raises(ValueError, match="indexer_budget"):
        QSASparseAttentionConfig().to_sparse_params(pretrained_config=SimpleNamespace())


@pytest.mark.parametrize(
    "field",
    ("index_n_heads", "index_kv_heads", "index_head_dim", "token_topk", "compress_ratio"),
)
def test_qsa_config_rejects_checkpoint_geometry_overrides(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        QSASparseAttentionConfig(**{field: 1})


def test_qsa_config_preserves_explicit_dense_threshold() -> None:
    sparse_config = QSASparseAttentionConfig(seq_len_threshold=16384)
    params = sparse_config.to_sparse_params(
        pretrained_config=SimpleNamespace(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
        )
    )

    assert sparse_config.seq_len_threshold == 16384
    assert params.seq_len_threshold == 16384
    assert params.dense_seq_len_threshold == 16384


def test_qsa_config_resolves_geometry_once_for_runtime_consumers() -> None:
    checkpoint_config = SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=1024,
        indexer_compress_ratio=4,
    )
    sparse_config = QSASparseAttentionConfig()
    resolved = sparse_config._resolve_checkpoint_defaults(checkpoint_config)

    assert sparse_config.seq_len_threshold is None
    assert resolved.seq_len_threshold == 1024
    assert resolved.to_sparse_params() == sparse_config.to_sparse_params(
        pretrained_config=checkpoint_config
    )


def test_model_config_resolves_qsa_geometry_before_graph_and_cache_setup() -> None:
    checkpoint_config = SimpleNamespace(
        indexer_n_heads=6,
        indexer_kv_heads=1,
        indexer_head_dim=96,
        indexer_budget=4096,
        indexer_compress_ratio=8,
    )

    model_config = ModelConfig(
        pretrained_config=checkpoint_config,
        sparse_attention_config=QSASparseAttentionConfig(),
    )

    assert model_config.sparse_attention_config == (
        QSASparseAttentionConfig()._resolve_checkpoint_defaults(checkpoint_config)
    )


def test_qsa_sparse_hook_is_registered() -> None:
    attention = SimpleNamespace(sparse_params=_sparse_params())

    hooks = get_sparse_attention_hooks(attention)

    assert isinstance(hooks, QSASparseHooks)


def test_qsa_empty_batch_keeps_the_regular_backend_path() -> None:
    metadata = object.__new__(QSAAttentionMetadata)
    metadata._num_tokens = 0

    output = QSASparseHooks().forward(
        attention=SimpleNamespace(),
        q=torch.empty((0,)),
        k=None,
        v=None,
        attn_metadata=metadata,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=None,
        attention_mask_data=None,
        mrope_config=None,
        attention_sinks=None,
        relative_attention_bias=None,
        relative_attention_max_distance=0,
        has_lora=False,
    )

    assert output is None


def test_qsa_metadata_allows_a_pp_rank_without_local_sparse_layers(monkeypatch) -> None:
    monkeypatch.setattr(TrtllmAttentionMetadata, "__post_init__", lambda self: None)
    manager = object.__new__(QSAMambaHybridCacheManagerV2)
    manager.qsa_position_layer_id = None
    metadata = object.__new__(QSAAttentionMetadata)
    metadata.kv_cache_manager = manager
    metadata.sparse_metadata_params = QSASparseMetadataParams(
        token_topk=2048,
        compress_ratio=4,
    )

    metadata.__post_init__()

    assert not metadata.qsa_has_local_layers


def test_qsa_hybrid_routes_to_sparse_v2_cache_manager(monkeypatch) -> None:
    from tensorrt_llm._torch.pyexecutor import _util

    monkeypatch.setattr(_util, "is_hybrid_linear", lambda config: True)
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        sparse_attention_config=QSASparseAttentionConfig(),
        get_num_mamba_layers=lambda: 1,
    )
    kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)

    manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)

    assert manager_cls is QSAMambaHybridCacheManagerV2


def test_qsa_hybrid_rejects_kv_cache_manager_v1(monkeypatch) -> None:
    from tensorrt_llm._torch.pyexecutor import _util

    monkeypatch.setattr(_util, "is_hybrid_linear", lambda config: True)
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        sparse_attention_config=QSASparseAttentionConfig(),
        get_num_mamba_layers=lambda: 1,
    )

    with pytest.raises(ValueError, match="requires use_kv_cache_manager_v2=True"):
        get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(use_kv_cache_manager_v2=False),
        )


def test_qsa_rejects_non_hybrid_models(monkeypatch) -> None:
    from tensorrt_llm._torch.pyexecutor import _util

    monkeypatch.setattr(_util, "is_hybrid_linear", lambda config: False)
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        sparse_attention_config=QSASparseAttentionConfig(),
    )

    with pytest.raises(ValueError, match="requires a hybrid"):
        get_kv_cache_manager_cls(
            model_config,
            KvCacheConfig(use_kv_cache_manager_v2=True),
        )


def test_qsa_cache_manager_uses_resolved_index_geometry(
    monkeypatch,
) -> None:
    from tensorrt_llm._torch.pyexecutor import _util

    checkpoint_config = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        num_hidden_layers=2,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=96,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    mamba_params = MambaKVCacheParams(
        state_size=128,
        conv_kernel=4,
        num_heads=16,
        n_groups=16,
        head_dim=128,
        mamba_layer_mask=[True, False],
        target_full_attention_layer_mask=[False, True],
        num_mamba_layers=1,
        num_draft_layers=0,
        dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(_util, "is_gemma4_hybrid", lambda config: False)
    monkeypatch.setattr(_util, "is_kimi_linear", lambda config: False)
    monkeypatch.setattr(_util, "is_mla", lambda config: False)
    monkeypatch.setattr(_util, "is_nemotron_hybrid", lambda config: False)
    monkeypatch.setattr(_util, "is_qwen3_hybrid", lambda config: True)
    monkeypatch.setattr(
        _util, "extract_mamba_kv_cache_params", lambda *args, **kwargs: mamba_params
    )
    monkeypatch.setattr(_util, "get_sm_version", lambda: 103)
    monkeypatch.setattr(_util, "is_gdn_replay_enabled", lambda: False)
    monkeypatch.setattr(MambaHybridCacheManagerV2, "__init__", lambda self, *args, **kwargs: None)

    manager = _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=QSAMambaHybridCacheManagerV2,
        mapping=SimpleNamespace(enable_attention_dp=False),
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
        tokens_per_block=128,
        max_seq_len=4096,
        max_batch_size=8,
        spec_config=None,
        sparse_attention_config=QSASparseAttentionConfig(),
        max_num_tokens=1024,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=SimpleNamespace(
            pretrained_config=checkpoint_config,
            quant_config=None,
        ),
        dtype=torch.bfloat16,
        is_draft=False,
    )

    assert manager.qsa_index_dim == 96
    assert manager.qsa_index_kv_heads == 1


def test_qsa_cache_manager_requires_sparse_config() -> None:
    with pytest.raises(ValueError, match="sparse_attention_config is required"):
        QSAMambaHybridCacheManagerV2(layer_mask=[True])


@pytest.mark.parametrize("dtype", (DataType.NVFP4, DataType.FLOAT))
def test_qsa_cache_manager_delegates_regular_kv_layout(
    monkeypatch: pytest.MonkeyPatch, dtype: DataType
) -> None:
    manager = object.__new__(QSAMambaHybridCacheManagerV2)
    manager.dtype = dtype
    sentinel = torch.empty(0, dtype=torch.int8)
    calls: list[tuple[object, int, str]] = []

    def regular_get_buffers(
        self: MambaHybridCacheManagerV2, layer_idx: int, kv_layout: str
    ) -> torch.Tensor:
        calls.append((self, layer_idx, kv_layout))
        return sentinel

    monkeypatch.setattr(MambaHybridCacheManagerV2, "get_buffers", regular_get_buffers)

    assert manager.get_buffers(7, "HND") is sentinel
    assert calls == [(manager, 7, "HND")]


def test_qsa_uses_regular_attention_for_scale_paged_kv_cache() -> None:
    metadata = object.__new__(QSAAttentionMetadata)
    metadata._num_tokens = 1
    metadata.kv_cache_manager = SimpleNamespace(dtype=DataType.NVFP4)
    indexer = SimpleNamespace(project_and_update_cache=lambda *args: torch.empty((1, 4, 128)))
    attention = SimpleNamespace(indexer=indexer, layer_idx=0)

    output = QSASparseHooks().forward(
        attention=attention,
        q=torch.empty((1,)),
        k=None,
        v=None,
        attn_metadata=metadata,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=None,
        attention_mask_data=None,
        mrope_config=None,
        attention_sinks=None,
        relative_attention_bias=None,
        relative_attention_max_distance=0,
        has_lora=False,
        qsa_index_hidden_states=torch.empty((1, 1)),
        qsa_position_ids=torch.zeros((1,), dtype=torch.int32),
    )

    assert output is None
