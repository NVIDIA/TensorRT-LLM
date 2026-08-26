# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention_backend.sparse.hooks import get_sparse_attention_hooks
from tensorrt_llm._torch.attention_backend.sparse.qsa import (
    QSAMambaHybridCacheManagerV2,
    QSASparseParams,
)
from tensorrt_llm._torch.attention_backend.sparse.qsa.module import QSASparseHooks
from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Attention
from tensorrt_llm._torch.models.modeling_qwen4_exp_attention import Qwen4ExpAttention
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, QSASparseAttentionConfig


def _sparse_params() -> QSASparseParams:
    return QSASparseParams(
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        token_topk=2048,
        compress_ratio=4,
    )


def test_qsa_config_uses_checkpoint_geometry() -> None:
    checkpoint_config = SimpleNamespace(
        indexer_n_heads=6,
        indexer_kv_heads=1,
        indexer_head_dim=96,
        indexer_budget=1024,
        indexer_compress_ratio=8,
    )

    params = QSASparseAttentionConfig().to_sparse_params(pretrained_config=checkpoint_config)

    assert params == QSASparseParams(
        index_n_heads=6,
        index_kv_heads=1,
        index_head_dim=96,
        token_topk=1024,
        compress_ratio=8,
    )


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


def test_qsa_sparse_hook_is_registered() -> None:
    attention = SimpleNamespace(sparse_params=_sparse_params())

    hooks = get_sparse_attention_hooks(attention)

    assert isinstance(hooks, QSASparseHooks)


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


def test_qsa_forward_passes_selector_inputs_to_runtime_hook(monkeypatch) -> None:
    captured = {}

    def fake_forward(self, **kwargs):
        del self
        captured.update(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(Qwen3Attention, "forward", fake_forward)
    attention = object.__new__(Qwen4ExpAttention)
    hidden_states = torch.randn(3, 8)
    position_ids = torch.arange(3)
    metadata = object()

    output = Qwen4ExpAttention.forward(
        attention,
        position_ids=position_ids,
        hidden_states=hidden_states,
        attn_metadata=metadata,
    )

    assert output is hidden_states
    assert captured["qsa_index_hidden_states"] is hidden_states
    assert captured["qsa_position_ids"] is position_ids
    assert captured["attn_metadata"] is metadata
