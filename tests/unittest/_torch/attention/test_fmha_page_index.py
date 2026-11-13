# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
from tensorrt_llm._torch.attention.backends.fmha.utils import get_multi_ctas_kv_counter_size
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX


def test_phased_fmha_uses_v2_page_index_upper_bound_directly() -> None:
    calls: list[tuple[int, object]] = []
    bounds = iter((97, 101))

    def get_page_index_upper_bound(local_layer_idx: int, role: object) -> int:
        calls.append((local_layer_idx, role))
        return next(bounds)

    manager = object.__new__(KVCacheManagerV2)
    manager.impl = SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound)
    fmha = object.__new__(TrtllmGenFmha)
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
def test_phased_fmha_uses_remaining_v1_selected_pool_extent(kv_factor: int) -> None:
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
    fmha = object.__new__(TrtllmGenFmha)
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
    fmha = object.__new__(TrtllmGenFmha)
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
    assert get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) < needed
    assert get_multi_ctas_kv_counter_size(num_heads, batch * beam, sm_count) >= needed


def test_multi_ctas_kv_counter_size_keeps_multi_processor_floor() -> None:
    num_heads, batch, sm_count = 6, 1, 148
    assert get_multi_ctas_kv_counter_size(num_heads, batch, sm_count) >= (
        sm_count * torch.int32.itemsize
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("local_layer_idx", [0, 1])
def test_cute_dsl_mla_v2_per_layer_indices_read_correct_pages(
    monkeypatch: pytest.MonkeyPatch, local_layer_idx: int
) -> None:
    """Per-layer indices and layer-origin buffers must select the same physical page."""
    page_size, latent_dim, rope_dim = 2, 2, 2
    layer_idx = 16 + local_layer_idx
    physical_pool = (
        torch.arange(10, dtype=torch.bfloat16)
        .view(10, 1, 1, 1, 1)
        .expand(-1, 1, page_size, 1, latent_dim + rope_dim)
        .contiguous()
    )
    # Two MLA target layers share a pool. Mixed target/draft layouts use
    # per-layer tables: slot 3 maps to physical pages 6 and 7 for these layers.
    block_offsets = torch.tensor(
        [[[[6, BAD_PAGE_INDEX]]], [[[7, BAD_PAGE_INDEX]]]], dtype=torch.int32
    )
    original_offsets = block_offsets.clone()
    # V2 wraps the shifted raw pointer as storage origin; a slice alone would
    # retain the full pool's storage and let an incorrect rebase hide the bug.
    layer_buffer = physical_pool[local_layer_idx:].clone()

    def get_buffers(requested_layer_idx: int) -> torch.Tensor:
        assert requested_layer_idx == layer_idx
        return layer_buffer

    def get_page_index_converter(requested_local_idx: int, role: object) -> SimpleNamespace:
        assert (requested_local_idx, role) == (local_layer_idx, Role.KEY)
        return SimpleNamespace(layer_offset=local_layer_idx)

    manager = object.__new__(KVCacheManagerV2)
    manager._use_per_layer_page_tables = True
    manager.layer_offsets = {16: 0, 17: 1}
    manager.get_buffers = get_buffers
    manager.impl = SimpleNamespace(get_page_index_converter=get_page_index_converter)
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        kv_cache_block_offsets=block_offsets,
        host_kv_cache_pool_mapping=torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        num_contexts=0,
        max_num_requests=1,
        helix_position_offsets=None,
    )
    attn = SimpleNamespace(
        layer_idx=layer_idx,
        get_local_layer_idx=lambda meta: meta.kv_cache_manager.layer_offsets[layer_idx],
        num_heads=1,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=latent_dim,
        q_scaling=1.0,
    )
    q = torch.zeros((1, 1, latent_dim + rope_dim), dtype=torch.bfloat16)
    output = torch.empty((1, 1, latent_dim), dtype=torch.bfloat16)
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=AttentionForwardArgs(
            output=output, attention_input_type=AttentionInputType.generation_only
        ),
        workspace=torch.empty(0, dtype=torch.uint8),
        sequence_lengths=torch.tensor([page_size], dtype=torch.int32),
        tokens_per_block=page_size,
        num_requests=1,
    )
    pages_read: list[torch.Tensor] = []

    def read_page(q_latent, q_rope, c_latent, c_rope, page_table, *args) -> None:
        assert page_table[1, 0].item() == BAD_PAGE_INDEX
        page = page_table[0, 0].item()
        pages_read.append(torch.cat((c_latent[:, :, page], c_rope[:, :, page]), dim=1))

    monkeypatch.setattr(
        torch.ops.trtllm, "cute_dsl_mla_decode_fp16_blackwell", read_page, raising=False
    )
    monkeypatch.setattr(CuteDslMlaFmha, "_required_workspace_size", staticmethod(lambda *args: 0))
    monkeypatch.setattr(CuteDslMlaFmha, "_split_kv_plan", staticmethod(lambda *args: None))
    fmha = object.__new__(CuteDslMlaFmha)
    fmha._run_mla_decode(q, output, params, torch.bfloat16)

    assert len(pages_read) == 1
    torch.testing.assert_close(pages_read[0], physical_pool[6 + local_layer_idx, 0, :, 0, :])
    torch.testing.assert_close(block_offsets, original_offsets)


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
        # Multi-token decode under helix is admitted only when the speculative
        # verify-group buffers are armed; unarmed is what this contract test
        # exercises.
        _helix_spec_tokens_valid=False,
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


def test_cute_dsl_mla_rejects_h96_mixed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generation slice of a mixed H96 batch must decline CuTe DSL."""
    q = torch.empty((1, 96 * 576), dtype=torch.bfloat16)
    attn = SimpleNamespace(
        num_heads=96,
        has_fp8_kv_cache=False,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_idx=0,
    )
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=1,
        beam_width=1,
        helix_position_offsets=None,
        tokens_per_block=64,
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda _layer_idx: torch.empty(0, dtype=torch.bfloat16)
        ),
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((1, 96 * 512), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.generation_only,
    )
    monkeypatch.setattr(
        AutoTuner, "get", classmethod(lambda _cls: SimpleNamespace(is_tuning_mode=False))
    )
    monkeypatch.setattr(
        CuteDslMlaFmha, "_kernel_can_implement", staticmethod(lambda *_args: (True, ""))
    )
    fmha = object.__new__(CuteDslMlaFmha)
    supported, reason = fmha._is_supported_with_reason(q, attn, metadata, forward_args)

    assert not supported
    assert reason == "CuTe DSL MLA FMHA only supports decode-only batches."


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
