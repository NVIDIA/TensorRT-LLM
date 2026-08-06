# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest

from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import (
    _KIMI_K3_MLA_GEN_BACKEND_ENV,
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


def test_select_kimi_k3_mla_generation_backend_uses_trtllm_gen_for_fp8_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_KIMI_K3_MLA_GEN_BACKEND_ENV, "cute-dsl")
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)

    assert _select_mla_generation_backend(quant_config) == "trtllm-gen"
