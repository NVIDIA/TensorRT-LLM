# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import (
    _KIMI_K3_MLA_GEN_BACKEND_DEFAULT,
    _KIMI_K3_MLA_GEN_BACKEND_ENV,
    _kimi_k3_mla_decode_backend_policy,
    _select_mla_generation_backend,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


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


def test_cute_dsl_mla_mixed_batch_supports_explicit_fp8_cute_dsl() -> None:
    """FP8 KV + explicitly selected CuTe-DSL (K3) has no other generation backend, so the
    generation slice of a mixed (IFB) batch must stay on CuTe-DSL."""
    attn = SimpleNamespace(
        layer_idx=3,
        num_heads=12,
        flashinfer_mla_backend="cute-dsl",
        mla_backend_policy=None,
    )
    metadata = SimpleNamespace(num_contexts=2, kv_cache_manager=SimpleNamespace(dtype=DataType.FP8))

    assert CuteDslMlaFmha._supports_mixed_batch(attn, metadata, 16, torch.float8_e4m3fn)
    # Not FP8 (BF16 KV keeps the old gate) or no dtype given: unchanged behaviour.
    assert not CuteDslMlaFmha._supports_mixed_batch(attn, metadata, 16, torch.bfloat16)
    assert not CuteDslMlaFmha._supports_mixed_batch(attn, metadata)
    # FP8 but another backend selected: unchanged behaviour.
    attn.flashinfer_mla_backend = "trtllm-gen"
    assert not CuteDslMlaFmha._supports_mixed_batch(attn, metadata, 16, torch.float8_e4m3fn)
