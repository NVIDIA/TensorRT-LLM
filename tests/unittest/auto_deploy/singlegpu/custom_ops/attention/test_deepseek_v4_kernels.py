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

"""Tests for standalone DeepSeek V4 attention kernel microfeatures."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_kernels import (
    FP8_E4M3_DTYPE,
    deepseek_v4_compressor_pool_norm_rope_ref,
    deepseek_v4_fp8_block_dequant_ref,
    deepseek_v4_indexer_q_rope_quant_ref,
    deepseek_v4_inverse_rope_fp8_output_quant_ref,
    deepseek_v4_kv_rmsnorm_rope_ref,
    deepseek_v4_local_window_topk_idxs,
    deepseek_v4_q_rmsnorm_rope_ref,
    deepseek_v4_sparse_attention_microkernel_ref,
)
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import e8m0_to_uint8, maybe_e8m0_to_fp32


def _freqs(batch_size: int, seq_len: int, rope_dim: int, device: str | torch.device = "cpu"):
    phases = torch.randn(batch_size, seq_len, rope_dim // 2, dtype=torch.float32, device=device)
    return torch.polar(torch.ones_like(phases), phases)


def _manual_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    out = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def _manual_apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    nope = x[..., : x.shape[-1] - rope_dim]
    rope = x[..., -rope_dim:]
    rope_complex = torch.view_as_complex(rope.float().reshape(*rope.shape[:-1], -1, 2))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if freqs.dim() == rope_complex.dim() - 1:
        freqs = freqs.unsqueeze(-2)
    rope_out = torch.view_as_real(rope_complex * freqs).flatten(-2).to(x.dtype)
    return torch.cat([nope, rope_out], dim=-1)


def test_q_rmsnorm_rope_matches_manual_reference():
    torch.manual_seed(1)
    batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 3, 4, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    weight = torch.linspace(0.5, 1.25, head_dim)
    freqs = _freqs(batch_size, seq_len, rope_dim)
    eps = 1e-6

    actual = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(q, weight, freqs, eps, rope_dim)
    expected = _manual_apply_rope(_manual_rms_norm(q, weight, eps), freqs, rope_dim)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual,
        deepseek_v4_q_rmsnorm_rope_ref(q, weight, freqs, eps, rope_dim),
        rtol=0,
        atol=0,
    )


def test_kv_rmsnorm_rope_cache_insert_quantizes_nope_and_preserves_rope():
    torch.manual_seed(2)
    batch_size, seq_len, head_dim, rope_dim = 2, 3, 10, 4
    kv = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float32) * 0.25
    weight = torch.linspace(0.75, 1.5, head_dim)
    freqs = _freqs(batch_size, seq_len, rope_dim)
    cache_indices = torch.tensor([4, 2, 0, 5, 3, 1], dtype=torch.int64)
    nope_dim = head_dim - rope_dim
    fp8_block_size = 4
    num_scale_blocks = (nope_dim + fp8_block_size - 1) // fp8_block_size

    nope_cache = torch.empty(batch_size * seq_len, nope_dim, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(batch_size * seq_len, rope_dim, dtype=torch.bfloat16)
    scale_cache = torch.empty(batch_size * seq_len, num_scale_blocks, dtype=torch.float8_e8m0fnu)

    kv_out, nope_fp8, nope_scale = (
        torch.ops.auto_deploy.torch_deepseek_v4_kv_rmsnorm_rope_cache_insert(
            kv,
            weight,
            freqs,
            cache_indices,
            nope_cache,
            rope_cache,
            scale_cache,
            1e-6,
            rope_dim,
            fp8_block_size,
        )
    )

    expected_kv = deepseek_v4_kv_rmsnorm_rope_ref(kv, weight, freqs, 1e-6, rope_dim)
    torch.testing.assert_close(kv_out, expected_kv, rtol=1e-6, atol=1e-6)

    flat_indices = cache_indices.reshape(-1)
    torch.testing.assert_close(
        nope_cache[flat_indices].view(torch.uint8),
        nope_fp8.reshape(-1, nope_dim).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rope_cache[flat_indices].float(),
        expected_kv[..., -rope_dim:].reshape(-1, rope_dim).to(torch.bfloat16).float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        e8m0_to_uint8(scale_cache[flat_indices]),
        e8m0_to_uint8(nope_scale.reshape(-1, num_scale_blocks)),
        rtol=0,
        atol=0,
    )

    dequant_nope = deepseek_v4_fp8_block_dequant_ref(
        nope_fp8, nope_scale, fp8_block_size, dtype=torch.float32
    )
    torch.testing.assert_close(
        dequant_nope,
        expected_kv[..., :nope_dim],
        rtol=0.25,
        atol=maybe_e8m0_to_fp32(nope_scale).max().item(),
    )


@pytest.mark.parametrize(
    "compress_ratio,overlap,seq_len",
    [
        (4, True, 8),
        (128, False, 256),
    ],
)
def test_compressor_pool_norm_rope_matches_reference(
    compress_ratio: int,
    overlap: bool,
    seq_len: int,
):
    torch.manual_seed(3 + compress_ratio)
    batch_size, head_dim, rope_dim = 2, 8, 4
    channels = 2 if overlap else 1
    kv = torch.randn(batch_size, seq_len, channels * head_dim, dtype=torch.float32) * 0.1
    gate = torch.randn_like(kv)
    ape = torch.randn(compress_ratio, channels * head_dim, dtype=torch.float32) * 0.01
    weight = torch.linspace(0.5, 1.25, head_dim)
    freqs = _freqs(batch_size, seq_len, rope_dim)

    actual = torch.ops.auto_deploy.torch_deepseek_v4_compressor_pool_norm_rope(
        kv, gate, ape, weight, freqs, 1e-6, rope_dim, compress_ratio, overlap
    )
    expected = deepseek_v4_compressor_pool_norm_rope_ref(
        kv, gate, ape, weight, freqs, 1e-6, rope_dim, compress_ratio, overlap
    )

    assert actual.shape == (batch_size, seq_len // compress_ratio, head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_indexer_q_rope_quant_uses_fp8_with_e8m0_scales():
    torch.manual_seed(5)
    batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 4, 3, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32) * 0.125
    freqs = _freqs(batch_size, seq_len, rope_dim)

    q_fp8, scale = torch.ops.auto_deploy.torch_deepseek_v4_indexer_q_rope_quant(
        q, freqs, rope_dim, 4, "fp8"
    )

    assert q_fp8.dtype == FP8_E4M3_DTYPE
    assert scale.dtype == torch.float8_e8m0fnu
    torch.testing.assert_close(
        e8m0_to_uint8(scale),
        e8m0_to_uint8(deepseek_v4_indexer_q_rope_quant_ref(q, freqs, rope_dim, 4)[1]),
        rtol=0,
        atol=0,
    )
    dequant = deepseek_v4_fp8_block_dequant_ref(q_fp8, scale, 4, dtype=torch.float32)
    expected = _manual_apply_rope(q, freqs, rope_dim)
    torch.testing.assert_close(
        dequant, expected, rtol=0.25, atol=maybe_e8m0_to_fp32(scale).max().item()
    )


def test_inverse_rope_output_quant_roundtrips_rope_before_fp8_error():
    torch.manual_seed(6)
    batch_size, seq_len, num_heads, head_dim, rope_dim = 1, 5, 2, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32) * 0.1
    freqs = _freqs(batch_size, seq_len, rope_dim)
    q_norm = _manual_rms_norm(q, None, 1e-6)
    q_rope = _manual_apply_rope(q_norm, freqs, rope_dim)

    out_fp8, scale = torch.ops.auto_deploy.torch_deepseek_v4_inverse_rope_fp8_output_quant(
        q_rope, freqs, rope_dim, 4
    )
    expected_fp8, expected_scale = deepseek_v4_inverse_rope_fp8_output_quant_ref(
        q_rope, freqs, rope_dim, 4
    )

    torch.testing.assert_close(
        out_fp8.view(torch.uint8), expected_fp8.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(e8m0_to_uint8(scale), e8m0_to_uint8(expected_scale), rtol=0, atol=0)
    dequant = deepseek_v4_fp8_block_dequant_ref(out_fp8, scale, 4, dtype=torch.float32)
    torch.testing.assert_close(
        dequant, q_norm, rtol=0.25, atol=maybe_e8m0_to_fp32(scale).max().item()
    )


def test_sparse_attention_microkernel_matches_source_op_assembly():
    torch.manual_seed(7)
    batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 6
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    local_kv = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float32)
    compressed_kv = torch.randn(batch_size, 3, head_dim, dtype=torch.float32)
    compressed_idxs = torch.tensor([[[0, -1], [0, 1], [1, 2], [2, -1]]], dtype=torch.int64)
    attn_sink = torch.tensor([0.25, -0.5], dtype=torch.float32)
    window_size = 3
    softmax_scale = 0.375

    actual = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_microkernel(
        q, local_kv, compressed_kv, compressed_idxs, attn_sink, window_size, softmax_scale
    )

    local_idxs = deepseek_v4_local_window_topk_idxs(window_size, batch_size, seq_len, q.device)
    compressed_offset = torch.where(
        compressed_idxs >= 0,
        compressed_idxs + local_kv.shape[1],
        compressed_idxs,
    )
    topk_idxs = torch.cat([local_idxs, compressed_offset], dim=-1)
    expected = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q, torch.cat([local_kv, compressed_kv], dim=1), attn_sink, topk_idxs, softmax_scale
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual,
        deepseek_v4_sparse_attention_microkernel_ref(
            q, local_kv, compressed_kv, compressed_idxs, attn_sink, window_size, softmax_scale
        ),
        rtol=0,
        atol=0,
    )


def test_fake_implementations_return_expected_shapes_and_dtypes():
    with FakeTensorMode():
        batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 3, 4, 8, 4
        q = torch.empty(batch_size, seq_len, num_heads, head_dim)
        kv = torch.empty(batch_size, seq_len, head_dim)
        freqs = torch.empty(batch_size, seq_len, rope_dim // 2, dtype=torch.complex64)
        cache_indices = torch.empty(batch_size * seq_len, dtype=torch.int64)
        nope_cache = torch.empty(batch_size * seq_len, head_dim - rope_dim, dtype=FP8_E4M3_DTYPE)
        rope_cache = torch.empty(batch_size * seq_len, rope_dim, dtype=torch.bfloat16)
        scale_cache = torch.empty(batch_size * seq_len, 1, dtype=torch.float8_e8m0fnu)

        q_out = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(
            q, None, freqs, 1e-6, rope_dim
        )
        kv_out, kv_fp8, kv_scale = (
            torch.ops.auto_deploy.torch_deepseek_v4_kv_rmsnorm_rope_cache_insert(
                kv,
                None,
                freqs,
                cache_indices,
                nope_cache,
                rope_cache,
                scale_cache,
                1e-6,
                rope_dim,
                128,
            )
        )
        indexer_fp8, indexer_scale = torch.ops.auto_deploy.torch_deepseek_v4_indexer_q_rope_quant(
            q, freqs, rope_dim, 128, "fp8"
        )

    assert q_out.shape == q.shape
    assert kv_out.shape == kv.shape
    assert kv_fp8.shape == (*kv.shape[:-1], head_dim - rope_dim)
    assert kv_scale.shape == (*kv.shape[:-1], 1)
    assert indexer_fp8.shape == q.shape
    assert indexer_scale.shape == (*q.shape[:-1], 1)
    assert kv_fp8.dtype == FP8_E4M3_DTYPE
    assert indexer_fp8.dtype == FP8_E4M3_DTYPE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_cuda_graph_replay_for_q_rmsnorm_rope():
    torch.manual_seed(8)
    device = torch.device("cuda")
    batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 4, 3, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    freqs = _freqs(batch_size, seq_len, rope_dim, device)
    weight = torch.linspace(0.75, 1.25, head_dim, device=device)

    for _ in range(3):
        torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(q, weight, freqs, 1e-6, rope_dim)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_out = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(
            q, weight, freqs, 1e-6, rope_dim
        )

    q.copy_(torch.randn_like(q))
    expected = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(
        q, weight, freqs, 1e-6, rope_dim
    )
    graph.replay()
    torch.testing.assert_close(replay_out, expected, rtol=1e-6, atol=1e-6)
