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
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
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
        output_gate=None,
    )

    assert output is None


@pytest.mark.parametrize(
    "num_contexts,max_kv_len,expected_dispatches,expect_sparse_output",
    ((1, 4, 0, False), (0, 8, 0, True), (1, 8, 1, True)),
    ids=("below-threshold", "decode", "exact-sparse-prefill"),
)
def test_qsa_sparse_prefill_dispatch_counter(
    monkeypatch: pytest.MonkeyPatch,
    num_contexts: int,
    max_kv_len: int,
    expected_dispatches: int,
    expect_sparse_output: bool,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels, module

    head_dim = 4
    num_heads = 4
    metadata = object.__new__(QSAAttentionMetadata)
    metadata._num_tokens = 1
    metadata._num_contexts = num_contexts
    metadata._seq_lens = torch.ones(1, dtype=torch.int32)
    metadata.kv_lens_runtime = torch.tensor([max_kv_len], dtype=torch.int32)
    metadata.qsa_req_idx_per_token = torch.zeros(1, dtype=torch.int32)
    metadata.qsa_logical_positions = torch.tensor([max_kv_len - 1], dtype=torch.int64)
    metadata.qsa_sequence_lengths = torch.tensor([max_kv_len], dtype=torch.int32)
    metadata.qsa_visible_blocks = torch.ones(1, dtype=torch.int32)
    metadata.qsa_topk_indices = torch.zeros((1, 7), dtype=torch.int32)
    metadata.qsa_topk_row_starts = torch.zeros(1, dtype=torch.int32)
    metadata.qsa_block_table = torch.arange(max_kv_len, dtype=torch.int32).unsqueeze(0)
    kv_pool = torch.zeros((max_kv_len, 2, 1, 1, head_dim))
    index_cache = torch.zeros((max_kv_len, 1, 1, head_dim))
    metadata.kv_cache_manager = SimpleNamespace(
        dtype=DataType.BF16,
        tokens_per_block=1,
        get_buffers=lambda *args, **kwargs: kv_pool,
        get_index_k_buffer=lambda *args, **kwargs: index_cache,
    )

    indexer = SimpleNamespace(
        project_and_update_cache=lambda *args, **kwargs: torch.zeros((1, num_heads, head_dim)),
        top_k=None,
    )
    attention = SimpleNamespace(
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=1,
        q_scaling=1.0,
        layer_idx=0,
        indexer=indexer,
        sparse_params=QSASparseParams(
            index_n_heads=num_heads,
            index_kv_heads=1,
            index_head_dim=head_dim,
            token_topk=4,
            compress_ratio=4,
            seq_len_threshold=4,
        ),
        split_qkv=lambda *args: (
            torch.zeros((1, num_heads, head_dim)),
            torch.zeros((1, 1, head_dim)),
            torch.zeros((1, 1, head_dim)),
        ),
    )
    monkeypatch.setattr(kernels, "qsa_supports_head_dims", lambda *args: True)
    monkeypatch.setattr(
        module,
        "select_qsa_paged_tokens",
        lambda *args, **kwargs: torch.zeros((1, 7), dtype=torch.int32),
    )
    monkeypatch.setattr(module, "qsa_sparse_gqa", lambda **kwargs: kwargs["q"])
    hooks = QSASparseHooks()

    output = hooks.forward(
        attention=attention,
        q=torch.zeros((1, head_dim)),
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
        output_gate=None,
        qsa_index_hidden_states=torch.zeros((1, head_dim)),
        qsa_position_ids=torch.zeros(1, dtype=torch.int32),
    )

    assert (output is not None) == expect_sparse_output
    assert hooks.num_sparse_prefill_dispatches == expected_dispatches


@pytest.mark.parametrize("in_graph", [False, True], ids=["padded-warmup", "capture"])
def test_qsa_fixed_output_bridge_captures_index_projection(
    monkeypatch: pytest.MonkeyPatch,
    in_graph: bool,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.qsa import custom_ops, module

    metadata = object.__new__(QSAAttentionMetadata)
    metadata._num_tokens = 2
    hidden_states = torch.randn(4, 16)
    position_ids = torch.arange(4, dtype=torch.int32)
    q_index = torch.randn(4, 4, 8)
    token_k = torch.randn(4, 1, 8)
    position_coordinates = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    projection_calls = []

    def project(hidden: torch.Tensor, positions: torch.Tensor):
        projection_calls.append((hidden, positions))
        return q_index, token_k, position_coordinates

    attention = SimpleNamespace(
        head_dim=8,
        indexer=SimpleNamespace(project=project),
        layer_idx_str="3",
        num_heads=4,
    )
    bridge_calls = []

    def bridge(*args: object) -> None:
        bridge_calls.append(args)
        output = args[-1]
        assert isinstance(output, torch.Tensor)
        output.fill_(7)

    monkeypatch.setattr(module, "is_in_breakable_cuda_graph", lambda: in_graph)
    monkeypatch.setattr(custom_ops, "maybe_bcg_qsa_attn_inplace", bridge)

    output = QSASparseHooks().forward(
        attention=attention,
        q=torch.empty((4, 64)),
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
        output_gate=None,
        qsa_index_hidden_states=hidden_states,
        qsa_position_ids=position_ids,
    )

    assert projection_calls == [(hidden_states, position_ids)]
    assert bridge_calls[0][3] is q_index
    assert bridge_calls[0][4] is token_k
    assert bridge_calls[0][5] is position_coordinates
    assert output.shape == (4, 32)
    torch.testing.assert_close(output, torch.full_like(output, 7))


def test_qsa_breakable_cuda_graph_bridge_uses_live_dense_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.qsa import custom_ops

    metadata = object()
    q_index = torch.randn(4, 4, 8)
    token_k = torch.randn(4, 1, 8)
    position_coordinates = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    calls = {}

    class _Hooks:
        def forward(self, *args: object, **kwargs: object):
            calls["metadata"] = args[4]
            calls["sparse_gate"] = args[13]
            calls["projection"] = kwargs["qsa_index_projection"]
            return None

    def dense_attention(*args: object, **kwargs: object):
        calls["dense_metadata"] = args[3]
        assert kwargs["output"].shape == (4, 32)
        return torch.full((2, 32), 5.0), None

    def apply_output_gate(output: torch.Tensor, gate: torch.Tensor):
        calls["dense_gate"] = gate
        return output + gate

    attention = SimpleNamespace(
        sparse_attn_hooks=_Hooks(),
        _attn_impl=dense_attention,
        apply_output_gate=apply_output_gate,
    )
    monkeypatch.setattr(
        custom_ops,
        "_extract_qsa_extra_attrs",
        lambda layer_idx: (metadata, attention),
    )
    output = torch.full((4, 32), float("nan"))
    output_gate = torch.full((4, 32), 2.0)

    custom_ops.qsa_attn_inplace(
        torch.empty((4, 64)),
        None,
        None,
        q_index,
        token_k,
        position_coordinates,
        None,
        None,
        output_gate,
        "3",
        output,
    )

    assert calls["metadata"] is metadata
    assert calls["dense_metadata"] is metadata
    assert calls["sparse_gate"] is output_gate
    torch.testing.assert_close(calls["dense_gate"], output_gate[:2])
    assert calls["projection"] == (q_index, token_k, position_coordinates)
    torch.testing.assert_close(output[:2], torch.full((2, 32), 7.0))
    torch.testing.assert_close(output[2:], torch.zeros((2, 32)))


def test_qsa_breakable_cuda_graph_bridge_copies_live_sparse_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.qsa import custom_ops

    metadata = object()
    q_index = torch.randn(4, 4, 8)
    token_k = torch.randn(4, 1, 8)
    position_coordinates = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    output_gate = torch.full((4, 32), 2.0)
    calls = {}

    class _Hooks:
        def forward(self, *args: object, **kwargs: object):
            calls["metadata"] = args[4]
            calls["gate"] = args[13]
            calls["projection"] = kwargs["qsa_index_projection"]
            return torch.full((2, 32), 6.0)

    def unexpected_dense_attention(*args: object, **kwargs: object):
        raise AssertionError("dense attention must not run after sparse dispatch")

    attention = SimpleNamespace(
        sparse_attn_hooks=_Hooks(),
        _attn_impl=unexpected_dense_attention,
    )
    monkeypatch.setattr(
        custom_ops,
        "_extract_qsa_extra_attrs",
        lambda layer_idx: (metadata, attention),
    )
    output = torch.full((4, 32), float("nan"))

    custom_ops.qsa_attn_inplace(
        torch.empty((4, 64)),
        None,
        None,
        q_index,
        token_k,
        position_coordinates,
        None,
        None,
        output_gate,
        "3",
        output,
    )

    assert calls["metadata"] is metadata
    assert calls["gate"] is output_gate
    assert calls["projection"] == (q_index, token_k, position_coordinates)
    torch.testing.assert_close(output[:2], torch.full((2, 32), 6.0))
    torch.testing.assert_close(output[2:], torch.zeros((2, 32)))


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


def test_qsa_uses_regular_attention_for_unsupported_head_dim() -> None:
    """Triton indexes head dims with tl.arange, so non-powers of two go dense."""
    metadata = object.__new__(QSAAttentionMetadata)
    metadata._num_tokens = 1
    metadata.kv_cache_manager = SimpleNamespace(dtype=DataType.BF16)
    # head_dim comes from the module, not from q: q is still the fused QKV here.
    attention = SimpleNamespace(head_dim=96, sparse_params=_sparse_params(), layer_idx=0)

    output = QSASparseHooks().forward(
        attention=attention,
        q=torch.empty((1, 96 * (4 + 2 + 2))),
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
        output_gate=None,
        qsa_index_hidden_states=torch.empty((1, 1)),
        qsa_position_ids=torch.zeros((1,), dtype=torch.int32),
    )

    assert output is None


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
        output_gate=None,
        qsa_index_hidden_states=torch.empty((1, 1)),
        qsa_position_ids=torch.zeros((1,), dtype=torch.int32),
    )

    assert output is None
