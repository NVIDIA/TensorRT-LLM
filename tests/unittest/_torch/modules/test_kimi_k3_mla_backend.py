# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Optional

import pytest

from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import (
    _KIMI_K3_MLA_GEN_BACKEND_ENV,
    _kimi_k3_mla_decode_backend_policy,
    _select_mla_generation_backend,
)
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


@pytest.mark.parametrize(
    (
        "requested_backend",
        "num_contexts",
        "num_generations",
        "num_gen_tokens",
        "num_heads",
        "expected_backend",
    ),
    [
        ("cute-dsl", 0, 4, 4, 96, "cute-dsl"),
        ("cute-dsl", 1, 3, 3, 12, "trtllm-gen"),
        ("cute-dsl", 1, 3, 3, 96, "cute-dsl"),
        ("cute-dsl", 0, 4, 8, 96, "trtllm-gen"),
        ("trtllm-gen", 1, 3, 3, 96, "trtllm-gen"),
    ],
)
def test_kimi_k3_mla_decode_backend_policy_by_batch_shape(
    requested_backend: str,
    num_contexts: int,
    num_generations: int,
    num_gen_tokens: int,
    num_heads: int,
    expected_backend: str,
) -> None:
    """K3 falls back outside plain decode except for unsafe H=96 mixed batches."""
    assert (
        _kimi_k3_mla_decode_backend_policy(
            requested_backend,
            SimpleNamespace(
                num_contexts=num_contexts,
                num_generations=num_generations,
            ),
            num_gen_tokens,
            num_heads=num_heads,
        )
        == expected_backend
    )
