# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for fused FLUX QK-norm + RoPE + static E4M3 quantization."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _generate_cos_sin(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    freqs = torch.randn(seq_len, head_dim // 2, device="cuda", dtype=torch.float32)
    freqs = freqs.repeat_interleave(2, dim=-1)
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _reference(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_add_weight: torch.Tensor | None,
    k_add_weight: torch.Tensor | None,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_txt_tokens: int,
    tokens_per_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_ref = qkv.clone()
    torch.ops.trtllm.fused_dit_qk_norm_rope(
        qkv_ref,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        cos,
        sin,
        num_txt_tokens,
        True,
        tokens_per_batch,
    )
    q, k, v = qkv_ref.split(num_heads * head_dim, dim=-1)
    outputs = []
    for operand, scale in zip((q, k, v), scales):
        quantized, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            operand.contiguous(), scale
        )
        outputs.append(quantized)
    return tuple(outputs)


def _fused(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_add_weight: torch.Tensor | None,
    k_add_weight: torch.Tensor | None,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_txt_tokens: int,
    tokens_per_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.trtllm.fused_dit_qk_norm_rope_quant_fp8(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        *scales,
        cos,
        sin,
        num_txt_tokens,
        True,
        tokens_per_batch,
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,num_txt_tokens",
    [(1, 65, -1), (1, 96, 24), (2, 65, 17)],
)
def test_fused_dit_qk_norm_rope_quant_fp8_parity(
    batch_size: int, seq_len: int, num_txt_tokens: int
) -> None:
    torch.manual_seed(7)
    num_heads, head_dim, eps = 4, 128, 1e-6
    num_tokens = batch_size * seq_len
    qkv = torch.randn(
        num_tokens,
        3 * num_heads * head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    q_add_weight = (
        torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") if num_txt_tokens > 0 else None
    )
    k_add_weight = (
        torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") if num_txt_tokens > 0 else None
    )
    scales = tuple(
        torch.tensor(value, dtype=torch.float32, device="cuda")
        for value in (0.0125, 0.01875, 0.025)
    )
    cos, sin = _generate_cos_sin(seq_len, head_dim)

    expected = _reference(
        qkv,
        num_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        scales,
        cos,
        sin,
        num_txt_tokens,
        seq_len,
    )
    actual = _fused(
        qkv,
        num_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        scales,
        cos,
        sin,
        num_txt_tokens,
        seq_len,
    )

    for name, fused, ref in zip(("Q", "K", "V"), actual, expected):
        assert fused.dtype == torch.float8_e4m3fn
        assert fused.is_contiguous()
        assert torch.equal(fused.view(torch.uint8), ref.view(torch.uint8)), (
            f"{name} differs from the existing DiT+static-quant reference"
        )


def test_fused_dit_qk_norm_rope_quant_fp8_cuda_graph() -> None:
    torch.manual_seed(19)
    num_heads, head_dim, seq_len = 4, 128, 64
    qkv = torch.randn(
        seq_len,
        3 * num_heads * head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    scales = tuple(
        torch.tensor(value, dtype=torch.float32, device="cuda") for value in (0.01, 0.015, 0.02)
    )
    cos, sin = _generate_cos_sin(seq_len, head_dim)
    args = (
        qkv,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        scales,
        cos,
        sin,
        -1,
        seq_len,
    )
    expected = _fused(*args)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _fused(*args)
    graph.replay()
    torch.cuda.synchronize()

    for fused, ref in zip(captured, expected):
        assert torch.equal(fused.view(torch.uint8), ref.view(torch.uint8))
