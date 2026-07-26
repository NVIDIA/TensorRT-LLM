# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha


class _AttentionStub:
    def __init__(
        self,
        *,
        is_mla_enable: bool,
        has_fp8_kv_cache: bool,
        flashinfer_mla_backend: str = "trtllm-gen",
    ) -> None:
        self.is_mla_enable = is_mla_enable
        self.has_fp8_kv_cache = has_fp8_kv_cache
        self.flashinfer_mla_backend = flashinfer_mla_backend
        self.kv_lora_rank = 512 if is_mla_enable else None
        self.head_dim = 576
        self.v_head_dim = 512 if is_mla_enable else None


def _get_total_num_blocks(manager: SimpleNamespace, kv_factor: int = 2) -> int:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = kv_factor
    return fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=manager))


def test_flashinfer_uses_v2_page_index_upper_bound_directly() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=50_000_000,
        impl=SimpleNamespace(get_page_index_upper_bound=lambda *_: 50_000_000),
        num_local_layers=36,
    )
    assert _get_total_num_blocks(manager) == 50_000_000


def test_flashinfer_preserves_legacy_pool_scaling() -> None:
    manager = SimpleNamespace(
        blocks_in_primary_pool=1024,
        impl=SimpleNamespace(),
        num_local_layers=36,
    )
    assert _get_total_num_blocks(manager, kv_factor=2) == 1024 * 36 * 2


def test_flashinfer_cute_dsl_mla_backend_rejects_fp8_kv_cache() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=True,
        flashinfer_mla_backend="cute-dsl",
    )

    with pytest.raises(ValueError, match="does not support FP8 KV cache"):
        FlashInferTrtllmGenFmha(attn)
