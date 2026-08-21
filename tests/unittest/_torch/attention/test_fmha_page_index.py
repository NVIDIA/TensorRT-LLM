# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_multi_ctas_kv_counter_size,
)


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


# The three tests below guard the MLA generation perf gate that #15300 removed as
# refactoring collateral, costing ~3% output token throughput on DeepSeek-V3-family
# and Kimi-K2 MLA decode at the default tokens_per_block (https://nvbugs/6617948).
# They deliberately call the checker instead of asserting on
# SLOWER_MLA_GENERATION_KERNELS itself: a test that pins the literal set would be
# deleted along with the constant by the next mechanical refactor, whereas these
# turn a dropped parameter into a TypeError and a dropped constant into an
# AttributeError.


def test_mla_generation_declines_slower_trtllm_gen_decode_kernel() -> None:
    # DeepSeek-V3 / Kimi-K2 shape at the default tokens_per_block=32: FlashInfer's
    # trtllm-gen MLA decode kernel is slower here than the thop.attention fallback,
    # so this backend must decline and let selection fall through.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=576,
        tokens_per_block=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert not supported
    assert "slower" in reason


@pytest.mark.parametrize("tokens_per_block", [16, 64])
def test_mla_generation_gate_is_scoped_to_one_page_size(tokens_per_block: int) -> None:
    # The gate must stay narrow: the same head dims at other page sizes are still
    # served by this backend. Real configs run tokens_per_block=64.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=576,
        tokens_per_block=tokens_per_block,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert supported, reason
    assert reason == ""


def test_mla_generation_allows_other_supported_head_dims() -> None:
    # (320, 256) is unaffected at every page size.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=320,
        tokens_per_block=32,
        kv_lora_rank=256,
        qk_rope_head_dim=64,
    )
    assert supported, reason
    assert reason == ""
