# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Optional

import pytest

from tensorrt_llm._torch.attention_backend.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.kimi_k3_cache_policy import (
    KIMI_K3_BF16_KV_LAYERS_ENV,
    get_kimi_k3_bf16_kv_layer_ids,
)
from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import (
    _KIMI_K3_MLA_GEN_BACKEND_DEFAULT,
    _KIMI_K3_MLA_GEN_BACKEND_ENV,
    _kimi_k3_mla_decode_backend_policy,
    _select_mla_generation_backend,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _kimi_k3_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="kimi_linear",
        num_hidden_layers=4,
        linear_attn_config={"full_attn_layers": [2, 4]},
    )


@pytest.mark.parametrize(
    "configured_layers, nvfp4_enabled, expected",
    [
        (None, True, {3, 7}),
        (None, False, set()),
        ("", True, set()),
        ("  ", True, set()),
        ("1", True, {1}),
        ("1", False, {1}),
    ],
)
@pytest.mark.parametrize("nested_text_config", [False, True])
def test_kimi_k3_bf16_kv_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
    configured_layers: Optional[str],
    nvfp4_enabled: bool,
    expected: set[int],
    nested_text_config: bool,
) -> None:
    if configured_layers is None:
        monkeypatch.delenv(KIMI_K3_BF16_KV_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(KIMI_K3_BF16_KV_LAYERS_ENV, configured_layers)
    config = SimpleNamespace(
        model_type="kimi_k3",
        num_hidden_layers=9,
        linear_attn_config={"full_attn_layers": [8, 2, 4]},
    )
    if nested_text_config:
        config = SimpleNamespace(text_config=config)

    assert get_kimi_k3_bf16_kv_layer_ids(config, nvfp4_enabled=nvfp4_enabled) == expected


@pytest.mark.parametrize("full_attn_layers, expected", [([], set()), ([2], {1}), ([2, 4], {1, 3})])
def test_kimi_k3_bf16_kv_default_with_fewer_mla_layers(
    monkeypatch: pytest.MonkeyPatch,
    full_attn_layers: list[int],
    expected: set[int],
) -> None:
    monkeypatch.delenv(KIMI_K3_BF16_KV_LAYERS_ENV, raising=False)
    config = _kimi_k3_config()
    config.linear_attn_config["full_attn_layers"] = full_attn_layers

    assert get_kimi_k3_bf16_kv_layer_ids(config, nvfp4_enabled=True) == expected


def test_kimi_k3_bf16_kv_fallback_parses_zero_based_mla_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KIMI_K3_BF16_KV_LAYERS_ENV, "1, 3")

    assert get_kimi_k3_bf16_kv_layer_ids(_kimi_k3_config()) == frozenset({1, 3})


@pytest.mark.parametrize("value", ["0", "4", "three", "1,"])
def test_kimi_k3_bf16_kv_fallback_rejects_invalid_layers(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(KIMI_K3_BF16_KV_LAYERS_ENV, value)

    with pytest.raises(ValueError, match=KIMI_K3_BF16_KV_LAYERS_ENV):
        get_kimi_k3_bf16_kv_layer_ids(_kimi_k3_config())


@pytest.mark.parametrize("configured_layers", [None, "1"])
def test_kimi_k3_bf16_kv_fallback_ignores_other_models(
    monkeypatch: pytest.MonkeyPatch,
    configured_layers: Optional[str],
) -> None:
    if configured_layers is None:
        monkeypatch.delenv(KIMI_K3_BF16_KV_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(KIMI_K3_BF16_KV_LAYERS_ENV, configured_layers)
    config = SimpleNamespace(
        model_type="deepseek_v3",
        num_hidden_layers=4,
        linear_attn_config={"full_attn_layers": [2, 4]},
    )

    assert get_kimi_k3_bf16_kv_layer_ids(config, nvfp4_enabled=True) == frozenset()


def test_kimi_k3_mla_generation_backend_default_is_trtllm_gen() -> None:
    # This branch diverges from upstream, which defaults to cute-dsl: the
    # shipped FlashInfer / CuTe-DSL pair cannot initialize the CuTe-DSL MLA
    # decode kernel, and every rank dies during executor init. Pinning the
    # default here makes that divergence fail loudly if someone reverts it
    # without also realigning the packages.
    assert _KIMI_K3_MLA_GEN_BACKEND_DEFAULT == "trtllm-gen"


@pytest.mark.parametrize(
    ("configured_backend", "expected_backend"),
    [
        (None, "trtllm-gen"),
        ("trtllm-gen", "trtllm-gen"),
        # The env var still opts back in, so the CuTe-DSL path stays reachable
        # wherever the FlashInfer and CuTe-DSL builds do agree.
        ("cute-dsl", "cute-dsl"),
    ],
)
def test_select_kimi_k3_mla_generation_backend(
    monkeypatch: pytest.MonkeyPatch,
    configured_backend: Optional[str],
    expected_backend: str,
) -> None:
    if configured_backend is None:
        monkeypatch.delenv(_KIMI_K3_MLA_GEN_BACKEND_ENV, raising=False)
    else:
        monkeypatch.setenv(_KIMI_K3_MLA_GEN_BACKEND_ENV, configured_backend)

    assert _select_mla_generation_backend(None) == expected_backend


@pytest.mark.parametrize("configured_backend", ["cute-dsl", "trtllm-gen"])
def test_select_kimi_k3_mla_generation_backend_requires_cute_dsl_for_fp8_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    configured_backend: str,
) -> None:
    monkeypatch.setenv(_KIMI_K3_MLA_GEN_BACKEND_ENV, configured_backend)
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)

    assert _select_mla_generation_backend(quant_config) == "cute-dsl"


@pytest.mark.parametrize(
    (
        "requested_backend",
        "num_contexts",
        "num_generations",
        "num_tokens",
        "expected_backend",
    ),
    [
        ("cute-dsl", 0, 4, 4, "cute-dsl"),
        ("cute-dsl", 1, 3, 4, "trtllm-gen"),
        ("cute-dsl", 0, 4, 8, "trtllm-gen"),
        ("trtllm-gen", 1, 3, 4, "trtllm-gen"),
    ],
)
def test_kimi_k3_mla_decode_backend_policy_falls_back_for_mixed_batches(
    requested_backend: str,
    num_contexts: int,
    num_generations: int,
    num_tokens: int,
    expected_backend: str,
) -> None:
    """K3's per-batch policy keeps CuTe-DSL only for generation-only,
    one-token-per-request batches; mixed or multi-token batches fall back
    to trtllm-gen."""
    assert (
        _kimi_k3_mla_decode_backend_policy(
            requested_backend,
            SimpleNamespace(
                num_contexts=num_contexts,
                num_generations=num_generations,
            ),
            num_tokens,
        )
        == expected_backend
    )


def test_cute_dsl_mla_mixed_batch_supports_bf16_layer_in_nvfp4_manager() -> None:
    manager = SimpleNamespace(
        dtype=DataType.NVFP4,
        layer_offsets={91: 91},
        get_kv_cache_dtype=lambda layer_idx: DataType.BF16,
    )
    attn = SimpleNamespace(layer_idx=91, num_heads=12)
    metadata = SimpleNamespace(num_contexts=9, kv_cache_manager=manager)

    assert CuteDslMlaFmha._supports_mixed_batch(attn, metadata)

    manager.get_kv_cache_dtype = lambda layer_idx: DataType.NVFP4
    assert not CuteDslMlaFmha._supports_mixed_batch(attn, metadata)

    metadata.kv_cache_manager = SimpleNamespace(
        dtype=DataType.BF16,
        layer_offsets={91: 91},
        get_kv_cache_dtype=lambda layer_idx: DataType.BF16,
    )
    assert not CuteDslMlaFmha._supports_mixed_batch(attn, metadata)

    metadata.num_contexts = 0
    assert CuteDslMlaFmha._supports_mixed_batch(attn, metadata)

    metadata.num_contexts = 9
    metadata.kv_cache_manager = None
    attn.num_heads = 96
    assert not CuteDslMlaFmha._supports_mixed_batch(attn, metadata)
