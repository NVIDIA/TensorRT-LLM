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


def _make_fmha(requested_backend: str, mla_backend_policy) -> FlashInferTrtllmGenFmha:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._mla_backend = requested_backend
    # ``Fmha.attn`` is a read-only property that dereferences ``_attn_ref``
    # (normally a weakref to the owning TrtllmAttention). SimpleNamespace is
    # not weak-referenceable, so stand in with a closure of the same shape.
    attn = SimpleNamespace(mla_backend_policy=mla_backend_policy)
    fmha._attn_ref = lambda: attn
    return fmha


@pytest.mark.parametrize("requested_backend", ["cute-dsl", "trtllm-gen"])
@pytest.mark.parametrize(
    ("num_contexts", "num_generations", "num_tokens"),
    [
        (0, 4, 4),  # generation-only, one token per request
        (1, 3, 4),  # mixed context/generation batch
        (0, 4, 8),  # multi-token generation (speculative verification)
    ],
)
def test_flashinfer_mla_backend_default_matches_static_selection(
    requested_backend: str,
    num_contexts: int,
    num_generations: int,
    num_tokens: int,
) -> None:
    """Without an installed policy the static backend is used for every batch
    composition, matching the behavior before the policy hook existed."""
    fmha = _make_fmha(requested_backend, mla_backend_policy=None)

    assert (
        fmha._get_effective_mla_backend(
            SimpleNamespace(
                num_contexts=num_contexts,
                num_generations=num_generations,
            ),
            num_tokens,
        )
        == requested_backend
    )


def test_flashinfer_mla_backend_policy_hook_is_consulted() -> None:
    calls = []

    def policy(requested_backend: str, meta, num_tokens: int) -> str:
        calls.append((requested_backend, meta, num_tokens))
        return "trtllm-gen"

    fmha = _make_fmha("cute-dsl", mla_backend_policy=policy)
    meta = SimpleNamespace(num_contexts=0, num_generations=4)

    assert fmha._get_effective_mla_backend(meta, 4) == "trtllm-gen"
    assert calls == [("cute-dsl", meta, 4)]
