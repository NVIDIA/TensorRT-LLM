# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.kimi_k3_mla import KimiK3MLAAttention
from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import (
    _KIMI_K3_MLA_GEN_BACKEND_ENV,
    _kimi_k3_mla_decode_backend_policy,
    _select_mla_generation_backend,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize(
    ("configured_backend", "expected_backend"),
    [(None, "cute-dsl"), ("trtllm-gen", "trtllm-gen")],
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


@pytest.mark.parametrize("invalid_backend", ["cutedsl", "", "CUTE-DSL "])
def test_select_kimi_k3_mla_generation_backend_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
    invalid_backend: str,
) -> None:
    """An invalid env value must fail at read time with a message naming the
    knob, not propagate until attention-backend construction."""
    monkeypatch.setenv(_KIMI_K3_MLA_GEN_BACKEND_ENV, invalid_backend)

    with pytest.raises(ValueError, match=_KIMI_K3_MLA_GEN_BACKEND_ENV):
        _select_mla_generation_backend(None)


def test_select_kimi_k3_mla_generation_backend_uses_trtllm_gen_for_fp8_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KIMI_K3_MLA_GEN_BACKEND_ENV, "cute-dsl")
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)

    assert _select_mla_generation_backend(quant_config) == "trtllm-gen"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="KimiK3MLAAttention builds real TRTLLM attention backends",
)
@pytest.mark.parametrize(
    ("kv_cache_quant_algo", "expected_backend"),
    [(None, "cute-dsl"), (QuantAlgo.FP8, "trtllm-gen")],
)
def test_kimi_k3_mla_construction_with_attention_dp_head_count(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_quant_algo: Optional[QuantAlgo],
    expected_backend: str,
) -> None:
    """Attention-DP keeps all 96 query heads on every rank and construction
    must succeed for both static backends. The FP8-KV override selects
    trtllm-gen; the FlashInfer wrapper later declines H=96 per batch and the
    legacy thop.attention TRTLLM-Gen path serves it. (The trtllm-gen variant
    was previously rejected in `__init__` because FlashInfer's TRTLLM-Gen MLA
    decode rejects 64 < num_heads_q < 128; CuTe-DSL has always run H=96.)
    """
    monkeypatch.delenv(_KIMI_K3_MLA_GEN_BACKEND_ENV, raising=False)
    model_config = ModelConfig(
        skip_create_weights_in_init=True,
        quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo),
        mapping=Mapping(world_size=4, tp_size=4, rank=0, enable_attention_dp=True),
    )

    # Kimi K3 MLA geometry (96 Q heads); small max positions keep the
    # identity-RoPE table allocation negligible.
    attn = KimiK3MLAAttention(
        hidden_size=7168,
        num_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        dtype=torch.bfloat16,
        layer_idx=0,
        max_position_embeddings=256,
        model_config=model_config,
    )

    assert attn.mqa.num_heads == 96
    assert attn.mqa.flashinfer_mla_backend == expected_backend
    assert attn.mqa.mla_backend_policy is _kimi_k3_mla_decode_backend_policy


@pytest.mark.parametrize(
    (
        "requested_backend",
        "num_contexts",
        "num_generations",
        "num_gen_tokens",
        "expected_backend",
    ),
    [
        ("cute-dsl", 0, 4, 4, "cute-dsl"),
        ("cute-dsl", 1, 3, 3, "trtllm-gen"),
        ("cute-dsl", 0, 4, 8, "trtllm-gen"),
        ("trtllm-gen", 1, 3, 3, "trtllm-gen"),
    ],
)
def test_kimi_k3_mla_decode_backend_policy_by_batch_shape(
    requested_backend: str,
    num_contexts: int,
    num_generations: int,
    num_gen_tokens: int,
    expected_backend: str,
) -> None:
    """K3 falls back to TRTLLM-Gen outside plain single-token decode."""
    assert (
        _kimi_k3_mla_decode_backend_policy(
            requested_backend,
            SimpleNamespace(
                num_contexts=num_contexts,
                num_generations=num_generations,
            ),
            num_gen_tokens,
        )
        == expected_backend
    )
