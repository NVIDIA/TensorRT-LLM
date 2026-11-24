# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 per-layer FP8 KV fallback under an NVFP4 (FP4 MLA) KV cache."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.kimi_k3_cache_policy import (
    KIMI_K3_FP8_KV_LAYERS_ENV,
    get_kimi_k3_fp8_kv_layer_ids,
    resolve_kimi_k3_mla_kv_cache_quant_config,
)
from tensorrt_llm.llmapi import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only

# Zero-based MLA layers 3, 7, 11; every other layer is KDA.
_MLA_LAYERS = (3, 7, 11)


def _kimi_k3_config(**overrides):
    config = dict(
        model_type="kimi_k3",
        num_hidden_layers=12,
        linear_attn_config={"full_attn_layers": [layer + 1 for layer in _MLA_LAYERS]},
    )
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    monkeypatch.delenv("TLLM_KIMI_K3_BF16_KV_LAYERS", raising=False)


def test_nvfp4_defaults_to_last_two_mla_layers() -> None:
    assert get_kimi_k3_fp8_kv_layer_ids(_kimi_k3_config(), nvfp4_enabled=True) == {7, 11}
    assert get_kimi_k3_fp8_kv_layer_ids(_kimi_k3_config(), nvfp4_enabled=False) == frozenset()


def test_vl_wrapper_uses_text_config() -> None:
    config = SimpleNamespace(text_config=_kimi_k3_config())
    assert get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True) == {7, 11}


def test_env_replaces_or_disables_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "3, 11")
    assert get_kimi_k3_fp8_kv_layer_ids(_kimi_k3_config(), nvfp4_enabled=True) == {3, 11}
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "")
    assert get_kimi_k3_fp8_kv_layer_ids(_kimi_k3_config(), nvfp4_enabled=True) == frozenset()


@pytest.mark.parametrize("value", ["4", "12", "3,,7", "x"])
def test_env_rejects_invalid_layers(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, value)
    with pytest.raises(ValueError, match=KIMI_K3_FP8_KV_LAYERS_ENV):
        get_kimi_k3_fp8_kv_layer_ids(_kimi_k3_config(), nvfp4_enabled=True)


def test_removed_bf16_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_KIMI_K3_BF16_KV_LAYERS", "11")
    with pytest.raises(ValueError, match=KIMI_K3_FP8_KV_LAYERS_ENV):
        get_kimi_k3_fp8_kv_layer_ids(_kimi_k3_config(), nvfp4_enabled=True)


def test_non_kimi_models_are_unaffected() -> None:
    config = _kimi_k3_config(model_type="deepseek_v3", architectures=["DeepseekV3ForCausalLM"])
    assert get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True) == frozenset()
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)
    assert resolve_kimi_k3_mla_kv_cache_quant_config(config, 11, quant_config) is quant_config


def test_resolve_switches_only_fallback_layers_to_fp8() -> None:
    config = _kimi_k3_config()
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES, kv_cache_quant_algo=QuantAlgo.NVFP4
    )
    # Populate the cached quant modes first: the fallback must not inherit them.
    assert quant_config.layer_quant_mode.has_fp4_kv_cache()

    assert resolve_kimi_k3_mla_kv_cache_quant_config(config, 3, quant_config) is quant_config
    resolved = resolve_kimi_k3_mla_kv_cache_quant_config(config, 11, quant_config)
    assert resolved is not quant_config
    assert resolved.kv_cache_quant_algo == QuantAlgo.FP8
    assert resolved.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert resolved.layer_quant_mode.has_fp8_kv_cache()
    assert not resolved.layer_quant_mode.has_fp4_kv_cache()
    assert quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
    # Idempotent: MLA resolves again in create_weights().
    assert resolve_kimi_k3_mla_kv_cache_quant_config(config, 11, resolved) is resolved


def test_resolve_rejects_explicit_selection_without_nvfp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "11")
    with pytest.raises(ValueError, match="NVFP4"):
        resolve_kimi_k3_mla_kv_cache_quant_config(_kimi_k3_config(), 11, QuantConfig())
    # Without the explicit selection a non-NVFP4 KV cache is left alone.
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV)
    quant_config = QuantConfig()
    assert resolve_kimi_k3_mla_kv_cache_quant_config(_kimi_k3_config(), 11, quant_config) is (
        quant_config
    )
