# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 V2 cache manager: the SM107 per-token estimate adds what the generic sum leaves out."""

import types

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import cache_manager

KVM = cache_manager.MiniMaxM3KVCacheManagerV2


def _model_config(nvfp4=True, index_dtype="fp8"):
    quant_mode = types.SimpleNamespace(
        has_fp4_kv_cache=lambda: nvfp4, has_fp8_kv_cache=lambda: not nvfp4
    )
    return types.SimpleNamespace(
        pretrained_config=types.SimpleNamespace(
            num_hidden_layers=60,
            sparse_attention_config={"sparse_attention_freq": [0] * 3 + [1] * 57},
            num_key_value_heads=4,
            head_dim=128,
            hidden_size=6144,
            num_attention_heads=64,
        ),
        sparse_attention_config=types.SimpleNamespace(
            sparse_index_dim=128, indexer_kv_dtype=index_dtype
        ),
        quant_config=types.SimpleNamespace(quant_mode=quant_mode),
    )


def _mapping(tp_size, attention_dp):
    return types.SimpleNamespace(tp_size=tp_size, pp_size=1, enable_attention_dp=attention_dp)


@pytest.fixture
def one_draft_layer(monkeypatch):
    monkeypatch.setattr(
        cache_manager, "shared_draft_layer_count", lambda spec_config, layer_mask: 1
    )


@pytest.mark.parametrize(
    "attention_dp,expected_extra",
    [
        # TP2 attention, 2 KV heads per rank: index-K 57 x 128 + dense 3 x (512 - 288) + draft 512.
        (False, 57 * 128 + 3 * (512 - 288) + 512),
        # Attention DP, 4 KV heads per rank: index-K 57 x 128 + dense 3 x (1024 - 576) + draft 1024.
        (True, 57 * 128 + 3 * (1024 - 576) + 1024),
    ],
)
def test_extra_bytes_per_token_nvfp4(one_draft_layer, attention_dp, expected_extra):
    extra = KVM._extra_bytes_per_token(
        _model_config(), _mapping(2, attention_dp), None, {"spec_config": object()}
    )
    assert extra == expected_extra


def test_separate_draft_cache_not_double_counted(one_draft_layer):
    kwargs = {"spec_config": object(), "use_separate_draft_kv_cache": True}
    extra = KVM._extra_bytes_per_token(_model_config(), _mapping(2, False), None, kwargs)
    assert extra == 57 * 128 + 3 * (512 - 288)


def test_estimate_gate(monkeypatch):
    calls = []

    def base(cls, model_config, mapping, num_layers=None, **kwargs):
        calls.append(kwargs)
        return (17280, 0)

    monkeypatch.setattr(
        cache_manager.KVCacheManagerV2, "get_cache_size_per_token", classmethod(base)
    )
    monkeypatch.setattr(KVM, "_extra_bytes_per_token", classmethod(lambda cls, *a: 8480))
    monkeypatch.setattr(cache_manager, "get_sm_version", lambda: 107)
    assert KVM.get_cache_size_per_token(None, None) == (25760, 0)
    assert KVM.get_cache_size_per_token(None, None, is_draft=True) == (17280, 0)
    monkeypatch.setattr(cache_manager, "get_sm_version", lambda: 103)
    assert KVM.get_cache_size_per_token(None, None) == (17280, 0)
    monkeypatch.setattr(cache_manager, "get_sm_version", lambda: 107)
    monkeypatch.setattr(cache_manager, "_SM107_EXACT_KV_SIZE_ESTIMATE", False)
    assert KVM.get_cache_size_per_token(None, None) == (17280, 0)


@pytest.mark.parametrize("dense_layers", [4, 10, 60])
@pytest.mark.parametrize("explicit", [False, True])
def test_estimate_matches_nondefault_cache_layout(monkeypatch, dense_layers, explicit):
    config = _model_config()
    sparse_ids = list(range(dense_layers, 60))
    kwargs = {}
    if explicit:
        kwargs["sparse_layer_ids"] = sparse_ids
    else:
        config.pretrained_config.sparse_attention_config = {
            "sparse_attention_freq": [0] * dense_layers + [1] * (60 - dense_layers)
        }
    manager = object.__new__(KVM)
    manager.sparse_layer_ids = (
        sparse_ids
        if explicit
        else cache_manager.default_sparse_layer_ids(config.pretrained_config, 60)
    )
    assert manager.sparse_layer_ids == sparse_ids
    manager.dtype = cache_manager.DataType.NVFP4
    manager.pp_layers = list(range(60))
    manager.layer_offsets = {layer: layer for layer in range(60)}
    manager.num_kv_heads_per_layer = [2] * 60
    manager.head_dim_per_layer = [128] * 60
    manager.kv_factor = 2
    manager.sparse_index_dim = 128
    manager.indexer_kv_dtype = "fp8"
    # The generic NVFP4 K/V allocation: 512 values / 2 + 512 scale bytes / 16.
    generic_layer_bytes = 288
    monkeypatch.setattr(
        cache_manager.KVCacheManagerV2,
        "get_layer_bytes_per_token",
        lambda self, layer, role: generic_layer_bytes,
    )
    # Role.ALL includes each sparse layer's index-K row, so it is the full allocation.
    layer_bytes = sum(
        manager.get_layer_bytes_per_token(i, cache_manager.Role.ALL) for i in range(60)
    )
    index_buffers = manager._extra_buffers_per_layer(tokens_per_block=128)
    index_bytes = (
        sum(buffer.size for buffers in index_buffers.values() for buffer in buffers) // 128
    )
    assert index_bytes == sum(
        manager.get_layer_bytes_per_token(i, cache_manager.Role.INDEX_KEY) for i in range(60)
    )
    extra = KVM._extra_bytes_per_token(config, _mapping(2, False), None, kwargs)
    assert 60 * generic_layer_bytes + extra == layer_bytes
