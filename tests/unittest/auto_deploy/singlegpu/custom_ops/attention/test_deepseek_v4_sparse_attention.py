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

import sys
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import Dim

_UTILS_DIR = Path(__file__).resolve().parents[3] / "_utils_test"
sys.path.append(str(_UTILS_DIR))
from _model_test_utils import assert_rmse_close  # noqa: E402

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: E402, F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo  # noqa: E402
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm  # noqa: E402
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
    logits = torch.einsum("bshd,bskd->bshk", q.to(compute_dtype), selected_kv)
    logits = logits * softmax_scale
    logits = logits.masked_fill((topk_idxs < 0).unsqueeze(2), float("-inf"))

    sink_logits = attn_sink.to(dtype=compute_dtype).view(1, 1, num_heads, 1)
    sink_logits = sink_logits.expand(batch_size, seq_len, num_heads, 1)
    weights = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    output = torch.einsum("bshk,bskd->bshd", weights[..., :-1], selected_kv)
    return output.to(q.dtype)


def _run_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float = 1.0,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q, kv, attn_sink, topk_idxs, softmax_scale
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
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        *metadata,
        swa_cache,
        softmax_scale,
        window_size,
        compress_ratio,
    )


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


def test_source_op_matches_reference_for_mixed_patterns() -> None:
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


def test_source_op_rejects_positive_topk_out_of_bounds() -> None:
    q = torch.randn(1, 1, 1, 2)
    kv = torch.randn(1, 2, 2)
    attn_sink = torch.randn(1)
    topk_idxs = torch.tensor([[[0, 2]]], dtype=torch.int64)

    with pytest.raises(ValueError, match="topk_idxs entries must be less than kv rows 2"):
        _run_sparse_attention(q, kv, attn_sink, topk_idxs)


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


def test_cached_ratio0_topk_decode_without_window_is_rejected() -> None:
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

    with pytest.raises(NotImplementedError, match="no unambiguous cache-position namespace"):
        _run_cached_sparse_attention(
            torch.tensor([[[[1.0, 0.0]]]]),
            torch.tensor([[[3.0, 0.0]]]),
            attn_sink,
            torch.zeros(1, 1, 1, dtype=torch.int64),
            _decode_meta(input_pos=1),
            swa_cache,
        )


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


def test_cached_sparse_attention_rejects_compressed_ratio() -> None:
    q = torch.randn(1, 1, 1, 2)
    kv = torch.randn(1, 1, 2)
    attn_sink = torch.randn(1)
    topk_idxs = torch.zeros(1, 1, 1, dtype=torch.int64)
    swa_cache = torch.empty(1, 4, 2)

    with pytest.raises(NotImplementedError, match="compress_ratio=0"):
        _run_cached_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            _decode_meta(input_pos=0),
            swa_cache,
            compress_ratio=4,
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


def test_cached_fake_tensor_rank_behavior() -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 2, 4)
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
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            1.0,
            False,
            "mha_sparse",
            0,
            4,
            self.compress_ratio,
            None,
            None,
            None,
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


def test_deepseek_sparse_cache_transform_rewrites_source_op_and_adds_resource() -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int64)
    gm = torch_export_to_gm(_TinyDeepSeekSparseModule(), (q, kv, attn_sink, topk_idxs))
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
    assert "r0_swa_cache" in placeholder_names
    assert "r0_swa_cache" in cm._resource_lookup


def test_deepseek_sparse_cache_transform_rejects_compressed_source_node() -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int64)
    gm = torch_export_to_gm(
        _TinyDeepSeekSparseModule(compress_ratio=4),
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

    with pytest.raises(RuntimeError, match="compressed sparse attention cache insertion"):
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
