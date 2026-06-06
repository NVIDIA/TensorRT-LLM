# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semantic tests for the DeepSeek V4 sparse attention source op."""

from __future__ import annotations

import pytest
import torch
from auto_deploy._utils_test._model_test_utils import assert_rmse_close
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: E402, F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (  # noqa: E402
    deepseek_v4_sparse_attention as dsv4_sparse,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo  # noqa: E402
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm  # noqa: E402
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (  # noqa: E402
    DeepseekV4Compressor,
    DeepseekV4Config,
    DeepseekV4Indexer,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface  # noqa: E402
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages  # noqa: E402
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (  # noqa: E402
    InsertCachedAttentionConfig,
    InsertCachedDeepSeekV4SparseAttention,
    _InsertCachedOperator,
)


def _context_meta(seq_len: int):
    batch_info_host = BatchInfo()
    batch_info_host.update([1, seq_len, 0, 0, 0, 0])
    return (
        batch_info_host.serialize(),
        torch.tensor([seq_len], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, seq_len], dtype=torch.int32),
    )


def _multi_context_meta(seq_lens: list[int]):
    total_tokens = sum(seq_lens)
    cu_seqlen = [0]
    for seq_len in seq_lens:
        cu_seqlen.append(cu_seqlen[-1] + seq_len)

    batch_info_host = BatchInfo()
    batch_info_host.update([len(seq_lens), total_tokens, 0, 0, 0, 0])
    return (
        batch_info_host.serialize(),
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.zeros(len(seq_lens), dtype=torch.int32),
        torch.arange(len(seq_lens), dtype=torch.int64),
        torch.tensor(cu_seqlen, dtype=torch.int32),
    )


def _decode_meta(input_pos: int):
    batch_info_host = BatchInfo()
    batch_info_host.update([0, 0, 0, 0, 1, 1])
    return (
        batch_info_host.serialize(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([input_pos], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int32),
    )


def _multi_decode_meta(input_positions: list[int]):
    seq_lens = [1] * len(input_positions)
    cu_seqlen = list(range(len(input_positions) + 1))
    batch_info_host = BatchInfo()
    batch_info_host.update([0, 0, 0, 0, len(input_positions), len(input_positions)])
    return (
        batch_info_host.serialize(),
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(input_positions, dtype=torch.int32),
        torch.arange(len(input_positions), dtype=torch.int64),
        torch.tensor(cu_seqlen, dtype=torch.int32),
    )


def _sparse_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, _ = q.shape
    batch_idx = torch.arange(batch_size, device=q.device).view(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, seq_len, topk_idxs.shape[-1])

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    gather_idxs = topk_idxs.to(torch.long).clamp(min=0)
    selected_kv = kv[batch_idx, gather_idxs].to(compute_dtype)
    logits = torch.matmul(q.to(compute_dtype), selected_kv.transpose(-1, -2))
    logits = logits * softmax_scale
    logits = logits.masked_fill((topk_idxs < 0).unsqueeze(2), float("-inf"))

    sink_logits = attn_sink.to(dtype=compute_dtype).view(1, 1, num_heads, 1)
    sink_logits = sink_logits.expand(batch_size, seq_len, num_heads, 1)
    weights = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    output = torch.matmul(weights[..., :-1], selected_kv)
    return output.to(q.dtype)


def _empty_sparse_attention_tensors(q: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, ...]:
    del kv
    return (
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(0, 0),
        q.new_empty(0),
        q.new_empty(0, 0),
        q.new_empty(0, 0),
        q.new_empty(q.shape[0], q.shape[1]),
        q.new_empty(q.shape[0], q.shape[1], 0, 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(0, 0),
        q.new_empty(0),
    )


def _run_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float = 1.0,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        *_empty_sparse_attention_tensors(q, kv),
        softmax_scale,
        compress_ratio=0,
    )


def _run_cached_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_cache: torch.Tensor,
    softmax_scale: float = 1.0,
    window_size: int | None = None,
    compress_ratio: int = 0,
) -> torch.Tensor:
    mhc_cache = swa_cache.new_empty(swa_cache.shape)
    compressor_kv_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    compressor_gate_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    indexer_compressor_kv_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    indexer_compressor_gate_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        *_empty_sparse_attention_tensors(q, kv),
        *metadata,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        None,
        1e-6,
        None,
    )


def _run_sparse_attention_with_compressor(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor: DeepseekV4Compressor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    softmax_scale: float = 1.0,
    window_size: int | None = None,
    compress_ratio: int = 0,
    indexer_q: torch.Tensor | None = None,
    indexer_weights: torch.Tensor | None = None,
    indexer_compressor_kv: torch.Tensor | None = None,
    indexer_compressor_gate: torch.Tensor | None = None,
    indexer_compressor_ape: torch.Tensor | None = None,
    indexer_compressor_norm_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    indexer_q = indexer_q if indexer_q is not None else q.new_empty(q.shape[0], q.shape[1], 0, 0)
    indexer_weights = (
        indexer_weights if indexer_weights is not None else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_kv = (
        indexer_compressor_kv
        if indexer_compressor_kv is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_gate = (
        indexer_compressor_gate
        if indexer_compressor_gate is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_ape = (
        indexer_compressor_ape if indexer_compressor_ape is not None else q.new_empty(0, 0)
    )
    indexer_compressor_norm_weight = (
        indexer_compressor_norm_weight
        if indexer_compressor_norm_weight is not None
        else q.new_empty(0)
    )
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor.ape,
        compressor.norm.weight,
        cos_table,
        sin_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        softmax_scale,
        False,
        "mha_sparse",
        0,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        kv.shape[-1],
        compressor.rope_head_dim,
        compressor.norm.eps,
    )


def _run_cached_sparse_attention_with_compressor(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor: DeepseekV4Compressor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float = 1.0,
    window_size: int = 4,
    compress_ratio: int = 4,
    indexer_q: torch.Tensor | None = None,
    indexer_weights: torch.Tensor | None = None,
    indexer_compressor_kv: torch.Tensor | None = None,
    indexer_compressor_gate: torch.Tensor | None = None,
    indexer_compressor_ape: torch.Tensor | None = None,
    indexer_compressor_norm_weight: torch.Tensor | None = None,
    indexer_compressor_kv_cache: torch.Tensor | None = None,
    indexer_compressor_gate_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    indexer_q = indexer_q if indexer_q is not None else q.new_empty(q.shape[0], q.shape[1], 0, 0)
    indexer_weights = (
        indexer_weights if indexer_weights is not None else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_kv = (
        indexer_compressor_kv
        if indexer_compressor_kv is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_gate = (
        indexer_compressor_gate
        if indexer_compressor_gate is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_ape = (
        indexer_compressor_ape if indexer_compressor_ape is not None else q.new_empty(0, 0)
    )
    indexer_compressor_norm_weight = (
        indexer_compressor_norm_weight
        if indexer_compressor_norm_weight is not None
        else q.new_empty(0)
    )
    indexer_compressor_kv_cache = (
        indexer_compressor_kv_cache
        if indexer_compressor_kv_cache is not None
        else q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    )
    indexer_compressor_gate_cache = (
        indexer_compressor_gate_cache
        if indexer_compressor_gate_cache is not None
        else q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    )
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor.ape,
        compressor.norm.weight,
        cos_table,
        sin_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        *metadata,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        compressor.norm.eps,
        compressor.rope_head_dim,
    )


def _rope_tables(max_seq_len: int, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_dim == 0:
        return torch.empty(max_seq_len, 0), torch.empty(max_seq_len, 0)
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.linspace(0.05, 0.25, rope_dim // 2, dtype=torch.float32).unsqueeze(0)
    angles = positions * freqs
    return angles.cos(), angles.sin()


def _compressor_case(
    compress_ratio: int,
    seq_len: int,
    *,
    compressed_capacity_tokens: int | None = None,
    batch_size: int = 1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    DeepseekV4Compressor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    hidden_size = 16
    head_dim = 8
    rope_dim = 4
    capacity = compressed_capacity_tokens or seq_len
    config = DeepseekV4Config(
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=capacity,
        ad_rope_cache_len=max(capacity, seq_len, 1),
    )
    compressor = DeepseekV4Compressor(config, compress_ratio, head_dim).eval()
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    compressor_kv, compressor_gate = compressor.project(hidden_states)
    cos_table, sin_table = _rope_tables(max(capacity, seq_len, 1), rope_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).contiguous()
    compressed_kv = compressor(hidden_states, cos_table, sin_table, position_ids)
    return (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    )


def _visible_source_topk(
    query_len: int,
    input_pos: int,
    kv_rows: int,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    max_select = window_size + max_compressed_len
    for token_offset in range(query_len):
        query_pos = input_pos + token_offset
        local_start = max(0, query_pos - window_size + 1)
        selected = list(range(local_start, query_pos + 1))
        visible_compressed = min((query_pos + 1) // compress_ratio, max_compressed_len)
        selected.extend(kv_rows + row_idx for row_idx in range(visible_compressed))
        selected.extend([-1] * (max_select - len(selected)))
        rows.append(selected)
    return torch.tensor([rows], dtype=torch.int64, device=device)


def _make_sparse_attention_caches(
    max_seq_len: int,
    head_dim: int,
    compressor_state_dim: int,
    fill_value: float = 0.0,
    *,
    num_slots: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((num_slots, max_seq_len, head_dim), fill_value),
        torch.full((num_slots, max_seq_len, head_dim), fill_value),
        torch.full((num_slots, max_seq_len, compressor_state_dim), fill_value),
        torch.full((num_slots, max_seq_len, compressor_state_dim), fill_value),
    )


def _has_resource_with_suffix(resource_names: list[str], suffix: str) -> bool:
    return any(name.endswith(suffix) for name in resource_names)


def test_sink_only_all_negative_topk_yields_finite_zero_output() -> None:
    q = torch.randn(1, 2, 2, 4)
    kv = torch.full((1, 5, 4), 1_000.0)
    attn_sink = torch.tensor([-3.0, 2.0])
    topk_idxs = torch.full((1, 2, 4), -1, dtype=torch.int64)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.5)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(q), rtol=0, atol=0)


def test_duplicate_topk_indices_preserve_independent_probability_mass() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0, 0, 1]]], dtype=torch.int64)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs)
    expected = _sparse_attention_reference(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="duplicate top-k: ")
    assert output[0, 0, 0, 0] > output[0, 0, 0, 1]


def test_negative_indices_are_masked_before_softmax() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[1000.0, 1000.0], [1.0, 0.0]]])
    attn_sink = torch.tensor([0.0])
    topk_idxs = torch.tensor([[[1, -1]]], dtype=torch.int32)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs)
    expected = _sparse_attention_reference(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="negative top-k mask: ")
    assert output.abs().max() < 1.0


def test_source_ratio0_matches_reference_for_mixed_patterns() -> None:
    torch.manual_seed(11)
    q = torch.randn(2, 4, 3, 6)
    kv = torch.randn(2, 8, 6)
    attn_sink = torch.tensor([-0.5, 0.25, 1.0])
    topk_idxs = torch.tensor(
        [
            [[0, 1, -1, 1], [2, 2, 3, -1], [4, -1, -1, 5], [6, 0, 6, 1]],
            [[7, 6, 5, 4], [3, -1, 3, 0], [-1, -1, -1, -1], [1, 2, 2, 7]],
        ],
        dtype=torch.int64,
    )

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.375)
    expected = _sparse_attention_reference(q, kv, attn_sink, topk_idxs, softmax_scale=0.375)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="mixed sparse attention: ")


def test_cached_ratio0_local_window_reads_past_kv_from_swa_cache() -> None:
    q_prefill = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    kv_prefill = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_prefill = torch.zeros(1, 3, 1, dtype=torch.int64)
    swa_cache = torch.empty(1, 8, 2)

    _run_cached_sparse_attention(
        q_prefill,
        kv_prefill,
        attn_sink,
        topk_prefill,
        _context_meta(seq_len=3),
        swa_cache,
        window_size=4,
    )

    q_decode = torch.tensor([[[[1.0, 0.5]]]])
    kv_decode = torch.tensor([[[3.0, -1.0]]])
    output = _run_cached_sparse_attention(
        q_decode,
        kv_decode,
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64),
        _decode_meta(input_pos=3),
        swa_cache,
        window_size=4,
    )

    expected_kv = torch.cat([kv_prefill, kv_decode], dim=1)
    expected_topk = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int64)
    expected = _sparse_attention_reference(q_decode, expected_kv, attn_sink, expected_topk, 1.0)

    torch.testing.assert_close(swa_cache[0, :4], expected_kv[0])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached local window: ")


def test_cached_ratio0_flattened_prefill_uses_per_sequence_kv_slice() -> None:
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]]]])
    kv = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 0.0], [0.0, 20.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0], [1], [0], [1]]], dtype=torch.int64)
    swa_cache = torch.empty(2, 8, 2)

    output = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _multi_context_meta([2, 2]),
        swa_cache,
        window_size=2,
    )

    expected_seq0 = _run_sparse_attention(q[:, :2], kv[:, :2], attn_sink, topk_idxs[:, :2])
    expected_seq1 = _run_sparse_attention(q[:, 2:], kv[:, 2:], attn_sink, topk_idxs[:, 2:])
    expected = torch.cat((expected_seq0, expected_seq1), dim=1)

    torch.testing.assert_close(swa_cache[0, :2], kv[0, :2])
    torch.testing.assert_close(swa_cache[1, :2], kv[0, 2:])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="flattened prefill: ")


def test_cached_ratio0_prefill_with_window_size_still_honors_topk_idxs() -> None:
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    kv = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int64)
    swa_cache = torch.empty(1, 8, 2)

    output = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _context_meta(seq_len=2),
        swa_cache,
        window_size=2,
    )
    expected = _run_sparse_attention(q, kv, attn_sink, topk_idxs)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached prefill topk: ")


def test_cached_ratio0_topk_mode_preserves_duplicates_and_negative_mask() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[2.0, 0.0], [100.0, 100.0], [1.0, 1.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0, 0, -1, 2]]], dtype=torch.int64)
    swa_cache = torch.empty(1, 8, 2)

    output = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _context_meta(seq_len=1),
        swa_cache,
    )
    expected = _run_sparse_attention(q, kv, attn_sink, topk_idxs)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached duplicate/mask: ")
    assert output[0, 0, 0, 0] > output[0, 0, 0, 1]


def test_cached_ratio0_topk_decode_without_window_uses_cache_positions() -> None:
    q_prefill = torch.zeros(1, 1, 1, 2)
    kv_prefill = torch.tensor([[[2.0, 0.0]]])
    attn_sink = torch.tensor([-20.0])
    swa_cache = torch.empty(1, 8, 2)

    _run_cached_sparse_attention(
        q_prefill,
        kv_prefill,
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64),
        _context_meta(seq_len=1),
        swa_cache,
    )

    q_decode = torch.tensor([[[[1.0, 0.0]]]])
    kv_decode = torch.tensor([[[3.0, 0.0]]])
    topk_decode = torch.tensor([[[0, 1]]], dtype=torch.int64)

    output = _run_cached_sparse_attention(
        q_decode,
        kv_decode,
        attn_sink,
        topk_decode,
        _decode_meta(input_pos=1),
        swa_cache,
    )

    expected_kv = torch.cat((kv_prefill, kv_decode), dim=1)
    expected = _sparse_attention_reference(q_decode, expected_kv, attn_sink, topk_decode, 1.0)

    torch.testing.assert_close(swa_cache[0, :2], expected_kv[0])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached top-k decode: ")


def test_cached_ratio0_sink_only_negative_topk_yields_zero_output() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[5.0, 5.0]]])
    attn_sink = torch.tensor([3.0])
    topk_idxs = torch.full((1, 1, 3), -1, dtype=torch.int64)
    swa_cache = torch.empty(1, 4, 2)

    output = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _decode_meta(input_pos=0),
        swa_cache,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(q), rtol=0, atol=0)


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_source_matches_expanded_sparse_construction(compress_ratio: int) -> None:
    torch.manual_seed(123 + compress_ratio)
    seq_len = compress_ratio
    q = torch.randn(1, seq_len, 1, 8)
    kv = torch.randn(1, seq_len, 8)
    attn_sink = torch.tensor([-0.25])
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len)
    topk_idxs = torch.arange(seq_len + compressor.max_compressed_len).view(1, 1, -1)
    topk_idxs = topk_idxs.expand(1, seq_len, -1).to(torch.int64)

    output = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        compress_ratio=compress_ratio,
    )
    expected = _run_sparse_attention(
        q,
        torch.cat((kv, compressed_kv), dim=1),
        attn_sink,
        topk_idxs,
    )

    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg=f"source ratio-{compress_ratio}: ",
    )


def test_cached_ratio0_prefill_and_decode_match_source() -> None:
    torch.manual_seed(37)
    total_len = 3
    prefill_len = 2
    q = torch.randn(1, total_len, 1, 8)
    kv = torch.randn(1, total_len, 8)
    attn_sink = torch.tensor([-0.25])
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(4, total_len)
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            total_len,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
        )
    )

    topk_prefill = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int64)
    output_prefill = _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=None,
        compress_ratio=0,
    )
    expected_prefill = _run_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        window_size=None,
        compress_ratio=0,
    )

    assert_rmse_close(
        output_prefill,
        expected_prefill,
        rmse_ratio_tol=1e-6,
        msg="cached ratio-0 prefill: ",
    )

    topk_decode = torch.tensor([[[0, 2]]], dtype=torch.int64)
    output_decode = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        topk_decode,
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=None,
        compress_ratio=0,
    )
    expected_decode = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        topk_decode,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=None,
        compress_ratio=0,
    )

    torch.testing.assert_close(swa_cache[0, :total_len], kv[0])
    assert_rmse_close(
        output_decode,
        expected_decode,
        rmse_ratio_tol=1e-6,
        msg="cached ratio-0 decode: ",
    )


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_cached_compressed_prefill_matches_source(compress_ratio: int) -> None:
    torch.manual_seed(311 + compress_ratio)
    seq_len = compress_ratio
    q = torch.randn(1, seq_len, 1, 8)
    kv = torch.randn(1, seq_len, 8)
    attn_sink = torch.tensor([-0.25])
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len)
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            seq_len,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
        )
    )
    topk_idxs = torch.arange(seq_len + compressor.max_compressed_len).view(1, 1, -1)
    topk_idxs = topk_idxs.expand(1, seq_len, -1).to(torch.int64)

    output = _run_cached_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        _context_meta(seq_len=seq_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        compress_ratio=compress_ratio,
    )
    expected = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        compress_ratio=compress_ratio,
    )

    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg=f"cached ratio-{compress_ratio} prefill: ",
    )


def test_cached_ratio128_token_input_pos_matches_full_source() -> None:
    torch.manual_seed(183)
    compress_ratio = 128
    total_len = 128
    compressed_capacity_tokens = 256
    prefill_len = total_len - 1
    window_size = 4
    q = torch.randn(1, total_len, 1, 8)
    kv = torch.randn(1, total_len, 8)
    attn_sink = torch.tensor([-0.5])
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=compressed_capacity_tokens,
    )
    state_dim = compressor_kv.shape[-1]
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            compressed_capacity_tokens,
            kv.shape[-1],
            state_dim,
            fill_value=777.0,
        )
    )

    topk_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    expected_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    torch.testing.assert_close(swa_cache[0, :total_len], kv[0])
    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg="cached token input_pos ratio-128: ",
    )


def test_cached_ratio128_multi_decode_metadata_matches_source_and_writes_slots() -> None:
    torch.manual_seed(217)
    batch_size = 2
    compress_ratio = 128
    total_len = 128
    prefill_len = total_len - 1
    window_size = 4
    compressed_capacity_tokens = 256
    q = torch.randn(batch_size, total_len, 1, 8)
    kv = torch.randn(batch_size, total_len, 8)
    attn_sink = torch.tensor([-0.5])
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=compressed_capacity_tokens,
        batch_size=batch_size,
    )
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            compressed_capacity_tokens,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
            num_slots=batch_size,
        )
    )

    topk_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    ).expand(batch_size, -1, -1)
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _multi_context_meta([prefill_len, prefill_len]),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    torch.testing.assert_close(mhc_cache[:, 0], torch.full_like(mhc_cache[:, 0], 777.0))

    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(batch_size, 1, 1, dtype=torch.int64),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _multi_decode_meta([prefill_len, prefill_len]),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    expected_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    ).expand(batch_size, -1, -1)
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    for slot_idx in range(batch_size):
        torch.testing.assert_close(swa_cache[slot_idx, :total_len], kv[slot_idx])
        torch.testing.assert_close(mhc_cache[slot_idx, 0], compressed_kv[slot_idx, 0])
        torch.testing.assert_close(
            mhc_cache[slot_idx, 1],
            torch.full_like(mhc_cache[slot_idx, 1], 777.0),
        )
    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg="cached ratio-128 multi decode: ",
    )


def test_cached_ratio4_decode_matches_source_with_learned_indexer_topk() -> None:
    torch.manual_seed(44)
    compress_ratio = 4
    prefill_len = 7
    total_len = 8
    window_size = 4
    q = torch.randn(1, total_len, 1, 8)
    kv = torch.randn(1, total_len, 8)
    attn_sink = torch.tensor([-0.5])
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, total_len)
    indexer_config = DeepseekV4Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        index_n_heads=1,
        index_head_dim=8,
        index_topk=1,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=total_len,
        ad_rope_cache_len=total_len,
    )
    indexer = DeepseekV4Indexer(indexer_config, compress_ratio).eval()
    hidden_states = torch.randn(1, total_len, indexer_config.hidden_size)
    q_lora = torch.randn(1, total_len, indexer_config.q_lora_rank)
    cos = cos_table[position_ids]
    sin = sin_table[position_ids]
    indexer_q, indexer_weights, indexer_compressor_kv, indexer_compressor_gate = indexer.project(
        hidden_states,
        q_lora,
        cos,
        sin,
    )
    compressed_idxs_prefill = indexer(
        hidden_states[:, :prefill_len],
        q_lora[:, :prefill_len],
        cos[:, :prefill_len],
        sin[:, :prefill_len],
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        prefill_len,
    )
    compressed_idxs_decode = indexer(
        hidden_states,
        q_lora,
        cos,
        sin,
        cos_table,
        sin_table,
        position_ids,
        total_len,
    )[:, prefill_len:]
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            8,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
        )
    )
    indexer_compressor_kv_cache = torch.full(
        (1, 8, indexer_compressor_kv.shape[-1]),
        777.0,
        dtype=indexer_compressor_kv.dtype,
    )
    indexer_compressor_gate_cache = torch.full_like(indexer_compressor_kv_cache, 777.0)

    local_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        0,
        q.device,
    )
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        torch.cat((local_prefill, compressed_idxs_prefill), dim=-1),
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        indexer_q=indexer_q[:, :prefill_len],
        indexer_weights=indexer_weights[:, :prefill_len],
        indexer_compressor_kv=indexer_compressor_kv[:, :prefill_len],
        indexer_compressor_gate=indexer_compressor_gate[:, :prefill_len],
        indexer_compressor_ape=indexer.compressor.ape,
        indexer_compressor_norm_weight=indexer.compressor.norm.weight,
        indexer_compressor_kv_cache=indexer_compressor_kv_cache,
        indexer_compressor_gate_cache=indexer_compressor_gate_cache,
    )
    torch.testing.assert_close(mhc_cache[0, 0], compressed_kv[0, 0])

    local_decode = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        0,
        q.device,
    )
    expected_topk = torch.cat((local_decode, compressed_idxs_decode), dim=-1)
    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, window_size + indexer.index_topk, dtype=torch.int64),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        indexer_q=indexer_q[:, prefill_len:],
        indexer_weights=indexer_weights[:, prefill_len:],
        indexer_compressor_kv=indexer_compressor_kv[:, prefill_len:],
        indexer_compressor_gate=indexer_compressor_gate[:, prefill_len:],
        indexer_compressor_ape=indexer.compressor.ape,
        indexer_compressor_norm_weight=indexer.compressor.norm.weight,
        indexer_compressor_kv_cache=indexer_compressor_kv_cache,
        indexer_compressor_gate_cache=indexer_compressor_gate_cache,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    all_visible_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    all_visible = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        all_visible_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    torch.testing.assert_close(mhc_cache[0, 1], compressed_kv[0, 1])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached ratio-4 indexer: ")
    assert not torch.allclose(output, all_visible, rtol=1e-6, atol=1e-6)


def test_cached_ratio4_indexer_runtime_all_reduce_flips_selected_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compress_ratio = 4
    window_size = 4
    max_seq_len = 12
    head_dim = 8
    index_head_dim = 2
    indexer_state_dim = 2 * index_head_dim
    input_pos = 8
    q = torch.zeros(1, 1, 1, head_dim)
    q[0, 0, 0, 0] = 1.0
    kv = torch.zeros(1, 1, head_dim)
    attn_sink = torch.tensor([-20.0])
    compressor_config = DeepseekV4Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=max_seq_len,
        ad_rope_cache_len=max_seq_len,
    )
    compressor = DeepseekV4Compressor(compressor_config, compress_ratio, head_dim).eval()
    compressor_kv = torch.zeros(1, 1, 2 * head_dim)
    compressor_gate = torch.zeros_like(compressor_kv)
    cos_table, sin_table = _rope_tables(max_seq_len, compressor.rope_head_dim)
    position_ids = torch.tensor([[input_pos]], dtype=torch.int64)
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            max_seq_len,
            head_dim,
            compressor_kv.shape[-1],
        )
    )
    mhc_cache[0, 0, 0] = 4.0
    mhc_cache[0, 1, 1] = 4.0

    indexer_q = torch.tensor([[[[0.0, 1.0]]]])
    indexer_weights = torch.ones(1, 1, 1)
    indexer_compressor_kv = torch.zeros(1, 1, indexer_state_dim)
    indexer_compressor_gate = torch.zeros_like(indexer_compressor_kv)
    indexer_compressor_kv_cache = torch.zeros(1, max_seq_len, indexer_state_dim)
    indexer_compressor_gate_cache = torch.zeros_like(indexer_compressor_kv_cache)
    indexer_compressor_kv_cache[0, :4, :index_head_dim] = torch.tensor([0.0, 1.0])
    indexer_compressor_kv_cache[0, :4, index_head_dim:] = torch.tensor([1.0, 0.0])
    indexer_compressor_kv_cache[0, 4:8, index_head_dim:] = torch.tensor([0.0, 1.0])
    indexer_compressor_ape = torch.zeros(compress_ratio, indexer_state_dim)
    indexer_compressor_norm_weight = torch.ones(index_head_dim)
    all_reduce_inputs: list[torch.Tensor] = []

    def fake_all_reduce(
        tensor: torch.Tensor,
        op: object = None,
        group: object = None,
        async_op: bool = False,
    ) -> None:
        assert op == dsv4_sparse.dist_common.ReduceOp.SUM
        assert group is None
        assert not async_op
        all_reduce_inputs.append(tensor.clone())
        tensor.copy_(torch.tensor([0.0, 100.0], dtype=tensor.dtype, device=tensor.device))

    monkeypatch.setattr(dsv4_sparse.dist_common, "is_initialized", lambda: True)
    monkeypatch.setattr(dsv4_sparse.dist_common, "get_world_size", lambda: 2)
    monkeypatch.setattr(dsv4_sparse.dist_common, "all_reduce", fake_all_reduce)

    output = _run_cached_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        torch.zeros(1, 1, window_size + 1, dtype=torch.int64),
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        _decode_meta(input_pos=input_pos),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_compressor_kv=indexer_compressor_kv,
        indexer_compressor_gate=indexer_compressor_gate,
        indexer_compressor_ape=indexer_compressor_ape,
        indexer_compressor_norm_weight=indexer_compressor_norm_weight,
        indexer_compressor_kv_cache=indexer_compressor_kv_cache,
        indexer_compressor_gate_cache=indexer_compressor_gate_cache,
    )

    assert len(all_reduce_inputs) == 1
    assert all_reduce_inputs[0][0] > all_reduce_inputs[0][1]
    expected_topk = torch.arange(window_size + 1, dtype=torch.int64).view(1, 1, -1)
    local_window = swa_cache[:, input_pos - window_size + 1 : input_pos + 1]
    expected = _run_sparse_attention(
        q,
        torch.cat((local_window, mhc_cache[:, 1:2]), dim=1),
        attn_sink,
        expected_topk,
    )
    without_reduce = _run_sparse_attention(
        q,
        torch.cat((local_window, mhc_cache[:, 0:1]), dim=1),
        attn_sink,
        expected_topk,
    )
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, without_reduce, rtol=1e-6, atol=1e-6)


def test_cached_ratio128_emits_boundary_row_and_hides_future_rows() -> None:
    torch.manual_seed(128)
    compress_ratio = 128
    total_len = 128
    prefill_len = total_len - 1
    window_size = 4
    q = torch.randn(1, total_len, 1, 8)
    kv = torch.randn(1, total_len, 8)
    attn_sink = torch.tensor([-0.5])
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=256,
    )
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            256,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
        )
    )

    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        _visible_source_topk(
            prefill_len,
            0,
            prefill_len,
            window_size,
            compress_ratio,
            compressor.max_compressed_len,
            q.device,
        ),
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    torch.testing.assert_close(mhc_cache[0, 0], torch.full_like(mhc_cache[0, 0], 777.0))

    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    expected_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    torch.testing.assert_close(mhc_cache[0, 0], compressed_kv[0, 0])
    torch.testing.assert_close(mhc_cache[0, 1], torch.full_like(mhc_cache[0, 1], 777.0))
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached ratio-128 boundary: ")


def test_cached_sparse_attention_rejects_unsupported_compress_ratio() -> None:
    q = torch.randn(1, 1, 1, 2)
    kv = torch.randn(1, 1, 2)
    attn_sink = torch.randn(1)
    topk_idxs = torch.zeros(1, 1, 1, dtype=torch.int64)
    swa_cache = torch.empty(1, 4, 2)

    with pytest.raises(ValueError, match="compress_ratio"):
        _run_cached_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            _decode_meta(input_pos=0),
            swa_cache,
            compress_ratio=2,
        )


def test_cached_sparse_attention_rejects_short_metadata() -> None:
    q = torch.randn(1, 1, 1, 2)
    kv = torch.randn(1, 1, 2)
    attn_sink = torch.randn(1)
    topk_idxs = torch.zeros(1, 1, 1, dtype=torch.int64)
    batch_info_host = BatchInfo()
    batch_info_host.update([1, 1, 0, 0, 0, 0])
    metadata = (
        batch_info_host.serialize(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="cu_seqlen must have at least 2 elements"):
        _run_cached_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            metadata,
            torch.empty(1, 4, 2),
        )


def test_fake_tensor_shape_behavior() -> None:
    q = torch.randn(2, 3, 2, 4)
    kv = torch.randn(2, 6, 4)
    attn_sink = torch.randn(2)
    topk_idxs = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 3]],
            [[3, 4], [4, 5], [5, -1]],
        ],
        dtype=torch.int64,
    )

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        q_fake = fake_mode.from_tensor(q)
        kv_fake = fake_mode.from_tensor(kv)
        sink_fake = fake_mode.from_tensor(attn_sink)
        topk_fake = fake_mode.from_tensor(topk_idxs)
        output = _run_sparse_attention(q_fake, kv_fake, sink_fake, topk_fake, softmax_scale=0.5)

    assert isinstance(output, FakeTensor)
    assert output.shape == q.shape
    assert output.dtype == q.dtype


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_cached_fake_tensor_rank_behavior(compress_ratio: int) -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 4 if compress_ratio else 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.zeros(1, 2, 1, dtype=torch.int64)
    metadata = _context_meta(seq_len=2)
    swa_cache = torch.empty(1, 8, 4)

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        q_fake = fake_mode.from_tensor(q)
        kv_fake = fake_mode.from_tensor(kv)
        sink_fake = fake_mode.from_tensor(attn_sink)
        topk_fake = fake_mode.from_tensor(topk_idxs)
        metadata_fake = tuple(fake_mode.from_tensor(tensor) for tensor in metadata)
        cache_fake = fake_mode.from_tensor(swa_cache)
        output = _run_cached_sparse_attention(
            q_fake,
            kv_fake,
            sink_fake,
            topk_fake,
            metadata_fake,
            cache_fake,
            window_size=2,
            compress_ratio=compress_ratio,
        )

    assert isinstance(output, FakeTensor)
    assert output.shape == q.shape
    assert output.dtype == q.dtype

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        with pytest.raises(ValueError, match="swa_cache must have rank 3"):
            _run_cached_sparse_attention(
                fake_mode.from_tensor(q),
                fake_mode.from_tensor(kv),
                fake_mode.from_tensor(attn_sink),
                fake_mode.from_tensor(topk_idxs),
                tuple(fake_mode.from_tensor(tensor) for tensor in metadata),
                fake_mode.from_tensor(torch.empty(1, 8, 1, 4)),
                window_size=2,
                compress_ratio=compress_ratio,
            )


def test_export_with_dynamic_batch_sequence_and_topk() -> None:
    class SparseAttentionModule(torch.nn.Module):
        def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_sink: torch.Tensor,
            topk_idxs: torch.Tensor,
        ) -> torch.Tensor:
            return _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.5)

    batch = Dim("batch", min=1, max=4)
    seq = Dim("seq", min=1, max=8)
    kv_rows = Dim("kv_rows", min=4, max=12)
    k_select = Dim("k_select", min=1, max=4)

    q = torch.randn(2, 3, 2, 4)
    kv = torch.randn(2, 6, 4)
    attn_sink = torch.randn(2)
    topk_idxs = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[5, 4], [3, 2], [1, 0]],
        ],
        dtype=torch.int64,
    )

    exported = torch.export.export(
        SparseAttentionModule(),
        (q, kv, attn_sink, topk_idxs),
        dynamic_shapes={
            "q": {0: batch, 1: seq},
            "kv": {0: batch, 1: kv_rows},
            "attn_sink": {},
            "topk_idxs": {0: batch, 1: seq, 2: k_select},
        },
    )

    target_names = {str(node.target) for node in exported.graph.nodes if node.op == "call_function"}
    assert "auto_deploy.torch_deepseek_v4_sparse_attention.default" in target_names

    q_alt = torch.randn(1, 4, 2, 4)
    kv_alt = torch.randn(1, 7, 4)
    sink_alt = torch.randn(2)
    topk_alt = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]])
    output = exported.module()(q_alt, kv_alt, sink_alt, topk_alt)
    assert output.shape == q_alt.shape


class _TinyDeepSeekSparseModule(torch.nn.Module):
    def __init__(self, compress_ratio: int = 0) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        state_dim = kv.shape[-1] * (2 if self.compress_ratio == 4 else 1)
        compressor_kv = q.new_empty(q.shape[0], q.shape[1], state_dim)
        compressor_gate = q.new_empty(q.shape[0], q.shape[1], state_dim)
        compressor_ape = q.new_empty(self.compress_ratio, state_dim)
        compressor_norm_weight = q.new_empty(kv.shape[-1])
        cos_table = q.new_empty(8, 0)
        sin_table = q.new_empty(8, 0)
        position_ids = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
        indexer_q = q.new_empty(q.shape[0], q.shape[1], 0, 0)
        indexer_weights = q.new_empty(q.shape[0], q.shape[1], 0)
        indexer_compressor_kv = q.new_empty(q.shape[0], q.shape[1], 0)
        indexer_compressor_gate = q.new_empty(q.shape[0], q.shape[1], 0)
        indexer_compressor_ape = q.new_empty(0, 0)
        indexer_compressor_norm_weight = q.new_empty(0)
        max_compressed_len = 2 if self.compress_ratio else None
        rope_dim = 0 if self.compress_ratio else None
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            position_ids,
            indexer_q,
            indexer_weights,
            indexer_compressor_kv,
            indexer_compressor_gate,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            1.0,
            False,
            "mha_sparse",
            0,
            4,
            self.compress_ratio,
            max_compressed_len,
            kv.shape[-1],
            rope_dim,
            1e-6,
        )


class _TinyDenseAttentionModule(torch.nn.Module):
    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_attention(
            qkv,
            qkv,
            qkv,
            None,
            0.0,
            True,
            1.0,
            None,
            None,
            None,
            "bsnd",
        )


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_deepseek_sparse_cache_transform_rewrites_source_op_and_adds_resource(
    compress_ratio: int,
) -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 4 if compress_ratio else 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int64)
    gm = torch_export_to_gm(
        _TinyDeepSeekSparseModule(compress_ratio=compress_ratio),
        (q, kv, attn_sink, topk_idxs),
    )
    cm = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=8,
        device="cpu",
    )

    transform = InsertCachedDeepSeekV4SparseAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT)
    )
    gm, info = transform._apply(gm, cm, factory=None, shared_config=SharedConfig())

    assert info.num_matches == 1
    targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    assert torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention.default not in targets
    assert torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache.default in targets

    placeholder_names = [node.target for node in gm.graph.nodes if node.op == "placeholder"]
    resource_names = list(cm._resource_lookup)
    for suffix in (
        "_swa_cache",
        "_mhc_cache",
        "_compressor_kv_cache",
        "_compressor_gate_cache",
    ):
        assert _has_resource_with_suffix(placeholder_names, suffix)
        assert _has_resource_with_suffix(resource_names, suffix)


def test_deepseek_sparse_cache_transform_rejects_unsupported_compress_ratio() -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int64)
    gm = torch_export_to_gm(
        _TinyDeepSeekSparseModule(compress_ratio=2),
        (q, kv, attn_sink, topk_idxs),
    )
    cm = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=8,
        device="cpu",
    )
    transform = InsertCachedDeepSeekV4SparseAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT)
    )

    with pytest.raises(RuntimeError, match="supports compress_ratio"):
        transform._apply(gm, cm, factory=None, shared_config=SharedConfig())


def test_dense_torch_attention_cache_insertion_remains_separate() -> None:
    qkv = torch.randn(1, 2, 1, 4)
    gm = torch_export_to_gm(_TinyDenseAttentionModule(), (qkv,))
    cm = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=8,
        device="cpu",
    )

    deepseek_transform = InsertCachedDeepSeekV4SparseAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT)
    )
    gm_after_deepseek, deepseek_info = deepseek_transform._apply(
        gm, cm, factory=None, shared_config=SharedConfig()
    )
    assert deepseek_info.num_matches == 0

    dense_transform = _InsertCachedOperator(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend="torch")
    )
    gm_after_dense, dense_info = dense_transform._apply(
        gm_after_deepseek, cm, factory=None, shared_config=SharedConfig()
    )

    assert dense_info.num_matches == 1
    targets = [node.target for node in gm_after_dense.graph.nodes if node.op == "call_function"]
    assert torch.ops.auto_deploy.torch_cached_attention_with_cache.default in targets
    assert (
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache.default not in targets
    )
