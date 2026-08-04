#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _reference_q_norm(q: torch.Tensor, head_dim: int, eps: float) -> torch.Tensor:
    q_view = q.view(-1, head_dim).float()
    inv_rms = torch.rsqrt(q_view.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (q_view * inv_rms).to(q.dtype).view_as(q)


@pytest.mark.parametrize("num_tokens", [1, 7, 129])
@pytest.mark.parametrize("num_heads", [1, 16])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="Requires BF16 support"
            ),
        ),
    ],
)
def test_deepseek_v4_q_norm_matches_reference(num_tokens, num_heads, dtype):
    torch.manual_seed(0)
    device = "cuda"
    head_dim = 512
    eps = 1e-6
    q = torch.randn(num_tokens, num_heads * head_dim, dtype=dtype, device=device).contiguous()

    out = torch.ops.trtllm.deepseek_v4_q_norm(q, num_heads, head_dim, eps)
    ref = _reference_q_norm(q, head_dim, eps)

    atol = 3.2e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 8e-3 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_deepseek_v4_q_norm_torch_compile_fullgraph():
    torch.manual_seed(0)
    num_heads = 16
    head_dim = 512
    eps = 1e-6
    q = torch.randn(8, num_heads * head_dim, dtype=torch.float16, device="cuda").contiguous()

    def q_norm(q):
        return torch.ops.trtllm.deepseek_v4_q_norm(q, num_heads, head_dim, eps)

    out = torch.compile(q_norm, backend="eager", fullgraph=True)(q)
    ref = _reference_q_norm(q, head_dim, eps)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_deepseek_v4_q_norm_rejects_invalid_inputs():
    num_heads = 2
    head_dim = 512
    eps = 1e-6
    q = torch.randn(3, num_heads * head_dim, dtype=torch.float16, device="cuda").contiguous()

    with pytest.raises(RuntimeError, match="CUDA|CPU|backend"):
        torch.ops.trtllm.deepseek_v4_q_norm(q.cpu(), num_heads, head_dim, eps)

    q_noncontiguous = torch.randn(num_heads * head_dim, 3, dtype=torch.float16, device="cuda").t()
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.trtllm.deepseek_v4_q_norm(q_noncontiguous, num_heads, head_dim, eps)

    with pytest.raises(RuntimeError, match="fp16/bf16"):
        torch.ops.trtllm.deepseek_v4_q_norm(q.float(), num_heads, head_dim, eps)

    with pytest.raises(RuntimeError, match="2D"):
        torch.ops.trtllm.deepseek_v4_q_norm(
            q.view(1, 3, num_heads * head_dim), num_heads, head_dim, eps
        )

    q_bad_shape = torch.randn(
        3, num_heads * head_dim + 1, dtype=torch.float16, device="cuda"
    ).contiguous()
    with pytest.raises(RuntimeError, match="num_heads \\* head_dim"):
        torch.ops.trtllm.deepseek_v4_q_norm(q_bad_shape, num_heads, head_dim, eps)

    with pytest.raises(RuntimeError, match="head_dim=512"):
        torch.ops.trtllm.deepseek_v4_q_norm(q, num_heads, 256, eps)
