# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which dense FP4 MLA configurations validate_fp4_mla_config accepts."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.config_utils import supports_fp4_mla_attention
from tensorrt_llm._torch.pyexecutor.model_loader import validate_fp4_mla_config

pytestmark = pytest.mark.cpu_only


def _model_config(*, cp_size: int = 1, helix: bool = False, pretrained_config=None):
    if pretrained_config is None:
        pretrained_config = SimpleNamespace(
            model_type="deepseek_v3", kv_lora_rank=512, qk_rope_head_dim=64
        )
    return SimpleNamespace(
        pretrained_config=pretrained_config,
        sparse_attention_config=None,
        attn_backend="TRTLLM",
        quant_config=SimpleNamespace(
            quant_mode=SimpleNamespace(has_fp4_kv_cache=lambda: True),
        ),
        mapping=SimpleNamespace(cp_size=cp_size, has_cp_helix=lambda: helix),
        spec_config=None,
    )


def _llm_args(*, use_kv_cache_manager_v2=None, enable_chunked_prefill: bool = False):
    return SimpleNamespace(
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=use_kv_cache_manager_v2),
        enable_chunked_prefill=enable_chunked_prefill,
    )


@pytest.mark.parametrize("enable_chunked_prefill", [False, True])
def test_chunked_prefill_is_accepted(enable_chunked_prefill: bool) -> None:
    validate_fp4_mla_config(
        _model_config(), _llm_args(enable_chunked_prefill=enable_chunked_prefill)
    )


@pytest.mark.parametrize("cp_size", [8, 16])
def test_helix_context_parallelism_is_accepted(cp_size: int) -> None:
    validate_fp4_mla_config(_model_config(cp_size=cp_size, helix=True), _llm_args())


def test_non_helix_context_parallelism_is_rejected() -> None:
    with pytest.raises(ValueError, match="only with Helix"):
        validate_fp4_mla_config(_model_config(cp_size=2, helix=False), _llm_args())


def test_kv_cache_manager_v1_is_rejected() -> None:
    with pytest.raises(ValueError, match="use_kv_cache_manager_v2=True"):
        validate_fp4_mla_config(_model_config(), _llm_args(use_kv_cache_manager_v2=False))


def test_non_trtllm_backend_is_not_dense_fp4_mla() -> None:
    model_config = _model_config()
    model_config.attn_backend = "FLASHINFER"
    assert not supports_fp4_mla_attention(model_config)
    with pytest.raises(ValueError, match="TRTLLM attention backend"):
        validate_fp4_mla_config(model_config, _llm_args())
