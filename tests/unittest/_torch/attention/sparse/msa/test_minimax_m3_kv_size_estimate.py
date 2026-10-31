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
