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

"""Tests for the DeepSeek V4 sparse attention source op."""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def _sparse_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, seq_len, _, _ = q.shape
    batch_idx = torch.arange(batch_size, device=q.device).reshape(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, seq_len, topk_idxs.shape[-1])

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    gather_idxs = topk_idxs.to(torch.long).clamp(min=0)
    selected_kv = kv[batch_idx, gather_idxs].to(compute_dtype)
    logits = torch.einsum("bshd,bskd->bshk", q.to(compute_dtype), selected_kv) * softmax_scale
    logits = logits.masked_fill((topk_idxs < 0).unsqueeze(2), float("-inf"))
    sink_logits = attn_sink.to(dtype=compute_dtype).reshape(1, 1, -1, 1)
    sink_logits = sink_logits.expand(batch_size, seq_len, q.shape[2], 1)
    weights_with_sink = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    return torch.einsum("bshk,bskd->bshd", weights_with_sink[..., :-1], selected_kv).to(q.dtype)


def _run_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float = 0.25,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q, kv, attn_sink, topk_idxs, softmax_scale
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_shape_and_dtype(dtype: torch.dtype):
    q = torch.randn(2, 3, 4, 5, dtype=dtype)
    kv = torch.randn(2, 7, 5, dtype=dtype)
    attn_sink = torch.randn(4, dtype=torch.float32)
    topk_idxs = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[4, 3, 2], [3, 2, 1], [2, 1, 0]],
        ],
        dtype=torch.int64,
    )

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs)

    assert output.shape == q.shape
    assert output.dtype == dtype
    assert output.is_contiguous()


@pytest.mark.parametrize(
    "topk_idxs",
    [
        pytest.param(
            torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]], dtype=torch.int64),
            id="local-sliding-window",
        ),
        pytest.param(
            torch.tensor([[[7, 2, 5], [6, 0, 4], [5, 1, 3], [4, 7, 2]]], dtype=torch.int64),
            id="compressed-nonlocal",
        ),
        pytest.param(
            torch.tensor([[[1, 1, 4], [2, 5, 2], [3, 3, 3], [6, 0, 6]]], dtype=torch.int32),
            id="duplicate-indices",
        ),
        pytest.param(
            torch.tensor([[[0, -1, -1], [1, 2, -1], [3, -1, 4], [5, 6, -1]]], dtype=torch.int32),
            id="masked-sentinel",
        ),
    ],
)
def test_reference_equality_for_index_patterns(topk_idxs: torch.Tensor):
    torch.manual_seed(7)
    q = torch.randn(1, 4, 3, 6, dtype=torch.float32)
    kv = torch.randn(1, 8, 6, dtype=torch.float32)
    attn_sink = torch.tensor([-0.5, 0.25, 1.0], dtype=torch.float32)
    softmax_scale = 0.375

    actual = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)
    expected = _sparse_attention_reference(q, kv, attn_sink, topk_idxs, softmax_scale)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_duplicate_indices_receive_independent_probability_mass():
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0, 0, 1]]], dtype=torch.int64)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    duplicate_logits = torch.tensor([2.0, 2.0, 0.0, -20.0])
    duplicate_weights = torch.softmax(duplicate_logits, dim=0)
    expected = duplicate_weights[0] * kv[:, 0] + duplicate_weights[1] * kv[:, 0]
    expected = expected + duplicate_weights[2] * kv[:, 1]
    torch.testing.assert_close(output[0, 0, 0], expected[0], rtol=1e-6, atol=1e-6)


def test_negative_indices_are_masked_before_softmax():
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[1.0, 0.0], [1000.0, 1000.0]]])
    attn_sink = torch.tensor([0.0])
    topk_idxs = torch.tensor([[[0, -1]]], dtype=torch.int64)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    weights = torch.softmax(torch.tensor([1.0, float("-inf"), 0.0]), dim=0)
    expected = weights[0] * kv[0, 0]
    torch.testing.assert_close(output[0, 0, 0], expected, rtol=1e-6, atol=1e-6)


def test_attention_sink_takes_probability_mass_without_value_vector():
    q = torch.zeros(1, 1, 1, 2)
    kv = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    topk_idxs = torch.tensor([[[0, 1]]], dtype=torch.int64)
    attn_sink = torch.tensor([0.0])

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    expected = torch.tensor([[[[2.0 / 3.0, 4.0 / 3.0]]]])
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)

    high_sink_output = _run_sparse_attention(
        q, kv, torch.tensor([10.0]), topk_idxs, softmax_scale=1.0
    )
    assert high_sink_output.norm() < output.norm()


def test_export_with_dynamic_batch_and_sequence():
    class SparseAttentionModule(torch.nn.Module):
        def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_sink: torch.Tensor,
            topk_idxs: torch.Tensor,
        ) -> torch.Tensor:
            return _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.5)

    batch = torch.export.Dim("batch", min=1, max=4)
    seq = torch.export.Dim("seq", min=1, max=8)
    kv_rows = torch.export.Dim("kv_rows", min=4, max=12)
    k_select = torch.export.Dim("k_select", min=1, max=4)

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

    assert any(
        "torch_deepseek_v4_sparse_attention" in str(node.target) for node in exported.graph.nodes
    )

    q_alt = torch.randn(1, 4, 2, 4)
    kv_alt = torch.randn(1, 7, 4)
    attn_sink_alt = torch.randn(2)
    topk_idxs_alt = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]])
    output = exported.module()(q_alt, kv_alt, attn_sink_alt, topk_idxs_alt)
    assert output.shape == q_alt.shape
