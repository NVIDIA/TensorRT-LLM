# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import (
    flashinfer_trtllm_gen as flashinfer_trtllm_gen_module,
)
from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_multi_ctas_kv_counter_size,
)
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


def test_flashinfer_uses_v2_page_index_upper_bound_directly() -> None:
    calls: list[tuple[int, object]] = []
    bounds = iter((97, 101))

    def get_page_index_upper_bound(local_layer_idx: int, role: object) -> int:
        calls.append((local_layer_idx, role))
        return next(bounds)

    manager = object.__new__(KVCacheManagerV2)
    manager.impl = SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound)
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    fmha._v1_total_num_blocks_cache = None
    attn = SimpleNamespace(local_layer_idx=7)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        host_kv_cache_pool_mapping=None,
    )

    assert fmha._get_total_num_blocks(metadata) == 97
    assert fmha._get_total_num_blocks(metadata) == 101
    assert calls == [(7, Role.KEY), (7, Role.KEY)]


@pytest.mark.parametrize("kv_factor", [1, 2])
def test_flashinfer_uses_remaining_v1_selected_pool_extent(kv_factor: int) -> None:
    calls: list[int] = []

    def get_primary_pool_data(local_layer_idx: int) -> SimpleNamespace:
        calls.append(local_layer_idx)
        return SimpleNamespace(shape=(1024,))

    pool_mapping = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 2],
        ],
        dtype=torch.int32,
    )
    manager = object.__new__(KVCacheManager)
    manager.impl = SimpleNamespace(get_primary_pool_data=get_primary_pool_data)
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = kv_factor
    fmha._v1_total_num_blocks_cache = None
    attn = SimpleNamespace(local_layer_idx=4)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        host_kv_cache_pool_mapping=pool_mapping,
    )

    expected = (1024 * 3 - 2) * kv_factor
    assert fmha._get_total_num_blocks(metadata) == expected
    assert fmha._get_total_num_blocks(metadata) == expected
    assert calls == [4]


def test_phased_fmha_rejects_unknown_kv_cache_manager_type() -> None:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    fmha._v1_total_num_blocks_cache = None
    fmha._attn_ref = lambda: SimpleNamespace(local_layer_idx=0)
    metadata = SimpleNamespace(kv_cache_manager=SimpleNamespace())

    with pytest.raises(TypeError, match="Unsupported KV cache manager: SimpleNamespace"):
        fmha._get_total_num_blocks(metadata)


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


@pytest.mark.parametrize("value, expected", [(None, 0), (0, 0), (8192, 8192)])
def test_attention_chunk_size_uses_zero_for_native_disabled_value(
    value: int | None,
    expected: int,
) -> None:
    assert FlashInferTrtllmGenFmha._get_attention_chunk_size(value) == expected


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
        "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen."
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
            params=SimpleNamespace(
                qkv_or_q=SimpleNamespace(),
                fwd=SimpleNamespace(),
                workspace=SimpleNamespace(),
            ),
            metadata=metadata,
        )


def test_flashinfer_generation_uses_phase_batch_size_for_padded_cross_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-attention padding can make active rows irrecoverable from requests and beam width."""
    batch_size = 3
    preprocess_calls: list[tuple[object, ...]] = []

    def generation_preprocess(*args: object) -> tuple[object, ...]:
        preprocess_calls.append(args)
        return (
            torch.empty((batch_size, 2, 4)),
            torch.empty(1),
            torch.empty((batch_size, 1), dtype=torch.int32),
            None,
            None,
            None,
            torch.empty(0, dtype=torch.uint8),
            None,
            1,
            1,
            -1,
            False,
        )

    decode_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        flashinfer_trtllm_gen_module.thop,
        "trtllm_gen_generation_preprocess",
        generation_preprocess,
    )
    monkeypatch.setattr(
        flashinfer_trtllm_gen_module,
        "flashinfer",
        SimpleNamespace(
            decode=SimpleNamespace(
                trtllm_batch_decode_with_kv_cache=lambda **kwargs: decode_calls.append(kwargs)
            )
        ),
        raising=False,
    )

    attn = SimpleNamespace(
        local_layer_idx=0,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        q_scaling=1.0,
        quant_mode=0,
        predicted_tokens_per_seq=1,
        attention_chunk_size=None,
        position_embedding_type=0,
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        rope_params=SimpleNamespace(dim=0, theta=1.0, scale_type=0, scale=1.0, max_positions=1),
    )
    metadata = SimpleNamespace(
        beam_width=2,
        kv_cache_block_offsets=torch.empty(0),
        host_kv_cache_pool_pointers=torch.empty(0),
        host_kv_cache_pool_mapping=torch.empty(0),
        num_contexts=2,
    )
    output = torch.empty((batch_size, 2, 4))
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.mixed,
    )
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(0, dtype=torch.uint8),
        output=output,
        qkv_or_q=torch.empty((batch_size, 2, 4)),
        sequence_length=torch.ones(batch_size, dtype=torch.int32),
        input_seq_length=1,
        num_tokens=batch_size,
        seq_offset=2,
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        head_size=attn.head_dim,
        q_scaling=attn.q_scaling,
        quant_mode=attn.quant_mode,
        predicted_tokens_per_seq=attn.predicted_tokens_per_seq,
        attention_chunk_size=attn.attention_chunk_size,
        position_embedding_type=attn.position_embedding_type,
        rope_params=attn.rope_params,
        tokens_per_block=32,
        kv_factor=2,
        total_num_blocks=8,
        num_seqs=batch_size,
        num_requests=1,
        beam_width=metadata.beam_width,
        is_cross=True,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._layout = "HND"
    fmha._enable_pdl = False
    fmha._multi_processor_count = 1
    fmha._multi_ctas_kv_counter_buffer = torch.empty(0, dtype=torch.int32)

    FlashInferTrtllmGenFmha.run_generation(fmha, params)

    assert preprocess_calls[0][24] == batch_size
    assert len(decode_calls) == 1


def _cute_dsl_mla_helix_support(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seq_len_q: int = 1,
    softmax_stats: torch.Tensor | None,
) -> tuple[bool, str]:
    batch_size, num_heads = 2, 96
    q = torch.empty(
        (batch_size * seq_len_q, num_heads * (512 + 64)),
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (batch_size * seq_len_q, num_heads * 512),
        dtype=torch.bfloat16,
    )
    attn = SimpleNamespace(
        num_heads=num_heads,
        has_fp8_kv_cache=False,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_idx=0,
    )
    metadata = SimpleNamespace(
        num_contexts=0,
        num_generations=batch_size,
        beam_width=1,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        helix_position_offsets=torch.zeros(batch_size, dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda _layer_idx: torch.empty(0, dtype=torch.bfloat16)
        ),
        tokens_per_block=64,
    )
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        softmax_stats_tensor=softmax_stats,
    )

    monkeypatch.setattr(
        AutoTuner,
        "get",
        classmethod(lambda _cls: SimpleNamespace(is_tuning_mode=False)),
    )
    monkeypatch.setattr(
        CuteDslMlaFmha,
        "_kernel_can_implement",
        staticmethod(lambda *_args: (True, "")),
    )
    fmha = object.__new__(CuteDslMlaFmha)
    return fmha._is_supported_with_reason(q, attn, metadata, forward_args)


def test_cute_dsl_mla_accepts_single_token_helix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = torch.empty((2, 96, 2), dtype=torch.float32)

    supported, reason = _cute_dsl_mla_helix_support(monkeypatch, softmax_stats=stats)

    assert supported, reason


@pytest.mark.parametrize(
    ("seq_len_q", "softmax_stats", "reason"),
    [
        (2, torch.empty((4, 96, 2), dtype=torch.float32), "single-token decode"),
        (1, None, "requires softmax_stats_tensor"),
        (1, torch.empty((2, 95, 2), dtype=torch.float32), "shape"),
        (1, torch.empty((2, 96, 2), dtype=torch.bfloat16), "float32"),
    ],
)
def test_cute_dsl_mla_rejects_invalid_helix_contract(
    monkeypatch: pytest.MonkeyPatch,
    seq_len_q: int,
    softmax_stats: torch.Tensor | None,
    reason: str,
) -> None:
    supported, actual_reason = _cute_dsl_mla_helix_support(
        monkeypatch,
        seq_len_q=seq_len_q,
        softmax_stats=softmax_stats,
    )

    assert not supported
    assert reason in actual_reason


# The tests below guard the MLA generation perf gate that #15300 removed as
# refactoring collateral, costing ~3% output token throughput on DeepSeek-V3-family
# and Kimi-K2 MLA decode at the default tokens_per_block. They deliberately call the
# checker instead of asserting on SLOWER_MLA_GENERATION_KERNELS itself: a test that
# pins the literal set would be deleted along with the constant by the next
# mechanical refactor, whereas these turn a dropped parameter into a TypeError and a
# dropped constant into an AttributeError.


def test_mla_generation_declines_slower_trtllm_gen_decode_kernel() -> None:
    # DeepSeek-V3 / Kimi-K2 shape at the default tokens_per_block=32: the trtllm-gen
    # MLA decode kernel is slower here than the thop.attention fallback, so this
    # backend must decline and let selection fall through.
    supported, reason = FlashInferTrtllmGenFmha._check_mla_generation_support(
        head_size=576,
        tokens_per_block=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert not supported
    assert "slower" in reason
    assert "headDimQk=576" in reason
    assert "headDimV=512" in reason
    assert "tokens_per_block=32" in reason


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
