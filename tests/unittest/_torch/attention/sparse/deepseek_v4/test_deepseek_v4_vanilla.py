# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4 Vanilla selected-attention golden."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
    DeepseekV4AttentionType,
    DeepSeekV4Params,
    DeepseekV4VanillaAttention,
)
from tensorrt_llm._torch.attention_backend.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.registry import (
    get_vanilla_sparse_attn_attention_backend,
)


class _FakeDeepseekV4CacheManager:
    def __init__(
        self,
        swa_cache: torch.Tensor,
        compressed_cache: torch.Tensor | None,
        tokens_per_block: int,
        compressed_tokens_per_block: int,
    ) -> None:
        self.swa_cache = swa_cache
        self.compressed_cache = compressed_cache
        self.tokens_per_block = tokens_per_block
        self.compressed_block_sizes = [compressed_tokens_per_block]
        self.layer_offsets = [0]

    def get_buffers(
        self,
        layer_idx: int,
        attention_type: DeepseekV4AttentionType,
    ) -> torch.Tensor:
        assert layer_idx == 0
        if attention_type == DeepseekV4AttentionType.SWA:
            return self.swa_cache
        if attention_type == DeepseekV4AttentionType.COMPRESS:
            assert self.compressed_cache is not None
            return self.compressed_cache
        raise ValueError(f"Unexpected attention type: {attention_type}")


def _create_backend(compress_ratio: int, window_size: int) -> DeepseekV4VanillaAttention:
    backend = object.__new__(DeepseekV4VanillaAttention)
    backend.layer_idx = 0
    backend.num_heads = 2
    backend.head_dim = 4
    backend.num_kv_heads = 1
    backend.quant_config = None
    backend.q_scaling = 1.0
    backend.qk_nope_head_dim = 2
    backend.qk_rope_head_dim = 2
    backend.v_head_dim = 4
    backend.compress_ratio = compress_ratio
    backend.window_size = window_size
    return backend


def _write_paged_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    rows: torch.Tensor,
    tokens_per_block: int,
) -> None:
    positions = torch.arange(rows.shape[0], dtype=torch.long)
    pages = block_table[torch.div(positions, tokens_per_block, rounding_mode="floor")]
    offsets = torch.remainder(positions, tokens_per_block)
    cache[pages, offsets] = rows


def _reference_attention(
    q: torch.Tensor,
    full_latent: torch.Tensor,
    compressed: torch.Tensor | None,
    topk_indices: torch.Tensor | None,
    *,
    past: int,
    compress_ratio: int,
    window_size: int,
    attention_sink: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    scale = 1.0 / math.sqrt(4)
    for token_idx in range(q.shape[0]):
        position = past + token_idx
        swa_start = max(0, position - window_size + 1)
        selected = [full_latent[swa_start : position + 1]]
        if compress_ratio == 4:
            assert compressed is not None and topk_indices is not None
            row = topk_indices[token_idx]
            valid = row[row >= 0].long()
            if valid.numel() > 0:
                selected.append(compressed.index_select(0, valid))
        elif compress_ratio == 128:
            assert compressed is not None
            num_compressed = (position + 1) // compress_ratio
            if num_compressed > 0:
                selected.append(compressed[:num_compressed])

        selected_latent = torch.cat(selected)
        scores = q[token_idx] @ selected_latent.transpose(0, 1) * scale
        scores = scores.float()
        sink = attention_sink.float().unsqueeze(1)
        max_score = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        numerator = torch.exp(scores - max_score)
        denominator = numerator.sum(dim=-1, keepdim=True)
        denominator += torch.exp(sink - max_score)
        probabilities = (numerator / denominator).to(q.dtype)
        outputs.append(probabilities @ selected_latent)
    return torch.stack(outputs).flatten(1)


@pytest.mark.parametrize(
    ("compress_ratio", "attention_input_type", "past", "q_len"),
    [
        (1, AttentionInputType.context_only, 0, 7),
        (4, AttentionInputType.context_only, 0, 9),
        (4, AttentionInputType.generation_only, 8, 2),
        (128, AttentionInputType.generation_only, 128, 2),
    ],
)
def test_deepseek_v4_vanilla_selected_attention(
    compress_ratio: int,
    attention_input_type: AttentionInputType,
    past: int,
    q_len: int,
) -> None:
    torch.manual_seed(41 + compress_ratio + past)
    window_size = 4
    tokens_per_block = 4
    compressed_tokens_per_block = 2
    total_length = past + q_len
    num_swa_pages = math.ceil(total_length / tokens_per_block)
    swa_cache = torch.zeros(num_swa_pages, tokens_per_block, 4)
    swa_block_table = torch.arange(num_swa_pages, dtype=torch.int32)

    past_latent = torch.randn(past, 4)
    new_latent = torch.randn(q_len, 4)
    _write_paged_rows(swa_cache, swa_block_table, past_latent, tokens_per_block)

    num_compressed = total_length // compress_ratio if compress_ratio > 1 else 0
    compressed = torch.randn(num_compressed, 4) if num_compressed > 0 else None
    compressed_cache = None
    compressed_block_table = None
    if compress_ratio > 1:
        num_compressed_pages = max(1, math.ceil(num_compressed / compressed_tokens_per_block))
        compressed_cache = torch.zeros(num_compressed_pages, compressed_tokens_per_block, 4)
        compressed_block_table = torch.arange(num_compressed_pages, dtype=torch.int32)
        if compressed is not None:
            _write_paged_rows(
                compressed_cache,
                compressed_block_table,
                compressed,
                compressed_tokens_per_block,
            )

    cache_manager = _FakeDeepseekV4CacheManager(
        swa_cache,
        compressed_cache,
        tokens_per_block,
        compressed_tokens_per_block,
    )
    sliding_block_tables = torch.full(
        (1, len(DeepseekV4AttentionType), 1, num_swa_pages),
        -1,
        dtype=torch.int32,
    )
    sliding_block_tables[0, DeepseekV4AttentionType.SWA.value, 0] = swa_block_table
    compress_block_tables = (
        {compress_ratio: compressed_block_table.unsqueeze(0)}
        if compressed_block_table is not None
        else {}
    )
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([q_len], dtype=torch.int32),
        num_contexts=1 if attention_input_type == AttentionInputType.context_only else 0,
        num_seqs=1,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[past]),
        kv_cache_manager=cache_manager,
        sliding_block_tables=sliding_block_tables,
        compress_block_tables=compress_block_tables,
        multi_item_part_lens=None,
    )

    topk_indices = None
    if compress_ratio == 4:
        topk_indices = torch.full((q_len, 3), -1, dtype=torch.int32)
        for token_idx in range(q_len):
            available = (past + token_idx + 1) // compress_ratio
            if available > 0:
                topk_indices[token_idx, 0] = 0
                topk_indices[token_idx, 1] = available - 1

    q = torch.randn(q_len, 2, 4)
    attention_sink = torch.tensor([0.25, -0.5])
    output = torch.empty(q_len, 8)
    backend = _create_backend(compress_ratio, window_size)
    result = backend.forward(
        q.flatten(1),
        None,
        None,
        metadata,
        forward_args=AttentionForwardArgs(
            output=output,
            latent_cache=new_latent,
            attention_sinks=attention_sink,
            attention_input_type=attention_input_type,
            sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk_indices),
        ),
    )

    expected = _reference_attention(
        q,
        torch.cat([past_latent, new_latent]),
        compressed,
        topk_indices,
        past=past,
        compress_ratio=compress_ratio,
        window_size=window_size,
        attention_sink=attention_sink,
    )
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result, expected)

    first_stored = max(past, total_length - window_size)
    stored_positions = torch.arange(first_stored, total_length, dtype=torch.long)
    stored_pages = swa_block_table[
        torch.div(stored_positions, tokens_per_block, rounding_mode="floor")
    ]
    stored_offsets = torch.remainder(stored_positions, tokens_per_block)
    actual_stored = swa_cache[stored_pages, stored_offsets]
    torch.testing.assert_close(actual_stored, new_latent[first_stored - past :])


def test_deepseek_v4_vanilla_rejects_future_compressed_index() -> None:
    backend = _create_backend(compress_ratio=4, window_size=4)
    swa_cache = torch.zeros(1, 4, 4)
    compressed_cache = torch.zeros(1, 2, 4)
    cache_manager = _FakeDeepseekV4CacheManager(swa_cache, compressed_cache, 4, 2)
    sliding_block_tables = torch.full(
        (1, len(DeepseekV4AttentionType), 1, 1),
        -1,
        dtype=torch.int32,
    )
    sliding_block_tables[0, DeepseekV4AttentionType.SWA.value, 0, 0] = 0
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        num_contexts=1,
        num_seqs=1,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        kv_cache_manager=cache_manager,
        sliding_block_tables=sliding_block_tables,
        compress_block_tables={4: torch.zeros(1, 1, dtype=torch.int32)},
        multi_item_part_lens=None,
    )

    with pytest.raises(ValueError, match="future compressed entry"):
        backend.forward(
            torch.randn(1, 8),
            None,
            None,
            metadata,
            forward_args=AttentionForwardArgs(
                latent_cache=torch.randn(1, 4),
                attention_input_type=AttentionInputType.context_only,
                sparse_backend_args=SparseBackendForwardArgs(
                    topk_indices=torch.tensor([[0]], dtype=torch.int32)
                ),
            ),
        )


def test_deepseek_v4_vanilla_backend_registry() -> None:
    sparse_params = DeepSeekV4Params(compress_ratios=[1])
    assert get_vanilla_sparse_attn_attention_backend(sparse_params) is DeepseekV4VanillaAttention


def test_deepseek_v4_vanilla_mla_rope_generation_accepts_full_contract() -> None:
    backend = _create_backend(compress_ratio=1, window_size=4)
    tensor = torch.empty(0)

    backend.mla_rope_generation(
        None,
        None,
        tensor,
        SimpleNamespace(),
        tensor,
        tensor,
        tensor,
        None,
        None,
        None,
        out_scale=tensor,
        kv_norm_weight=tensor,
        kv_norm_eps=1e-5,
        precomputed_cu_seqlens=True,
        precomputed_fmha_scheduler=True,
        kv_only=True,
        kv_done_elsewhere=False,
        quant_scale_qkv=tensor,
    )
