# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace
from typing import TypeAlias

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_multi_ctas_kv_counter_size,
)
from tensorrt_llm._torch.attention_backend.fmha.interface import _CuteDslMlaStagingKey
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata


class _AttentionStub:
    def __init__(
        self,
        *,
        is_mla_enable: bool,
        has_fp8_kv_cache: bool,
        flashinfer_mla_backend: str | None = None,
    ) -> None:
        self.is_mla_enable = is_mla_enable
        self.has_fp8_kv_cache = has_fp8_kv_cache
        self.flashinfer_mla_backend = flashinfer_mla_backend
        self.kv_lora_rank = 512 if is_mla_enable else None
        self.head_dim = 576
        self.v_head_dim = 512 if is_mla_enable else None


_MlaBackendPolicy: TypeAlias = Callable[[str, SimpleNamespace, int], str]


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


def test_multi_ctas_kv_counter_size_covers_beam_expanded_batch() -> None:
    # The kernel keeps one counter per head per decoder sequence. Sizing off the
    # request count alone under-allocates under beam search, but only once the
    # product clears the multi-processor floor, so pick a case that does.
    num_heads, batch, beam, sm_count = 6, 16, 2, 148
    needed = num_heads * batch * beam * torch.int32.itemsize
    assert _get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) < needed
    assert _get_multi_ctas_kv_counter_size(num_heads, batch * beam, sm_count) >= needed


def test_multi_ctas_kv_counter_size_keeps_multi_processor_floor() -> None:
    num_heads, batch, sm_count = 6, 1, 148
    assert _get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) >= (
        sm_count * torch.int32.itemsize
    )


def test_prepare_workspace_sizes_counter_for_max_num_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_heads, max_num_requests, beam_width, sm_count = 6, 16, 2, 148
    max_num_sequences = max_num_requests * beam_width

    def check_counter_size_args(
        actual_num_heads: int,
        actual_max_num_sequences: int,
        actual_sm_count: int,
    ) -> int:
        assert (actual_num_heads, actual_max_num_sequences, actual_sm_count) == (
            num_heads,
            max_num_sequences,
            sm_count,
        )
        raise RuntimeError("counter size arguments observed")

    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen."
        "_get_multi_ctas_kv_counter_size",
        check_counter_size_args,
    )

    fmha = SimpleNamespace(
        attn=SimpleNamespace(num_heads=num_heads),
        _multi_processor_count=sm_count,
    )
    metadata = SimpleNamespace(
        max_num_requests=max_num_requests,
        beam_width=beam_width,
        max_num_sequences=max_num_sequences,
    )
    with pytest.raises(RuntimeError, match="counter size arguments observed"):
        FlashInferTrtllmGenFmha.prepare_workspace(
            fmha,
            q=SimpleNamespace(),
            k=None,
            v=None,
            metadata=metadata,
            forward_args=SimpleNamespace(),
            workspace=SimpleNamespace(),
        )


def test_flashinfer_cute_dsl_mla_backend_rejects_fp8_kv_cache() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=True,
        flashinfer_mla_backend="cute-dsl",
    )

    with pytest.raises(ValueError, match="does not support FP8 KV cache"):
        FlashInferTrtllmGenFmha(attn)


@pytest.mark.parametrize("configured_backend", ["cute-dsl", "trtllm-gen"])
def test_standalone_cute_dsl_mla_defers_to_explicit_flashinfer_backend(
    configured_backend: str,
) -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=False,
        flashinfer_mla_backend=configured_backend,
    )

    assert not CuteDslMlaFmha.is_available(attn)


def test_flashinfer_mla_backend_defaults_to_trtllm_gen() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=False,
    )

    assert FlashInferTrtllmGenFmha(attn)._mla_backend == "trtllm-gen"


def test_mla_scheduler_invalidation_resets_cute_dsl_staging_key() -> None:
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata._mla_scheduler_buffers_valid = True
    metadata._mla_ctx_cu_seqlens_valid = True
    metadata._cute_dsl_mla_staging_key = _CuteDslMlaStagingKey(
        is_capturing=True,
        workspace_ptr=1,
        block_tables_ptr=2,
        block_tables_shape=(3, 4),
        sequence_lengths_ptr=5,
        sequence_lengths_offset=6,
        batch_beam=7,
        padded_num_pages=8,
    )

    metadata._invalidate_mla_scheduler_buffers()

    assert not metadata._mla_scheduler_buffers_valid
    assert not metadata._mla_ctx_cu_seqlens_valid
    assert metadata._cute_dsl_mla_staging_key is None


def test_flashinfer_mla_backend_rejects_unknown_backend() -> None:
    attn = _AttentionStub(
        is_mla_enable=True,
        has_fp8_kv_cache=False,
        flashinfer_mla_backend="cutedsl",
    )

    with pytest.raises(ValueError, match="flashinfer_mla_backend must be one of"):
        FlashInferTrtllmGenFmha(attn)


def _make_fmha(
    requested_backend: str,
    mla_backend_policy: _MlaBackendPolicy | None,
) -> FlashInferTrtllmGenFmha:
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
    ("num_contexts", "num_generations", "num_gen_tokens"),
    [
        (0, 4, 4),  # generation-only, one token per request
        (1, 3, 3),  # mixed context/generation batch
        (0, 4, 8),  # multi-token generation (speculative verification)
    ],
)
def test_flashinfer_mla_backend_default_matches_static_selection(
    requested_backend: str,
    num_contexts: int,
    num_generations: int,
    num_gen_tokens: int,
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
            num_gen_tokens,
        )
        == requested_backend
    )


def test_flashinfer_mla_backend_policy_hook_is_consulted() -> None:
    calls: list[tuple[str, SimpleNamespace, int]] = []

    def policy(
        requested_backend: str,
        meta: SimpleNamespace,
        num_gen_tokens: int,
    ) -> str:
        calls.append((requested_backend, meta, num_gen_tokens))
        return "trtllm-gen"

    fmha = _make_fmha("cute-dsl", mla_backend_policy=policy)
    meta = SimpleNamespace(num_contexts=0, num_generations=4)

    assert fmha._get_effective_mla_backend(meta, 4) == "trtllm-gen"
    assert calls == [("cute-dsl", meta, 4)]
