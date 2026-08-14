# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for fused DSpark RMSNorm and RoPE."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE or not is_sm_100f(),
    reason="DSpark fused RMSNorm+RoPE requires an SM100-family CUDA GPU",
)


def _make_inputs(
    batch: int,
    seq: int,
    hidden_dim: int,
    rope_dim: int,
    num_heads: int,
    seed: int = 0,
):
    torch.manual_seed(seed)
    shape = (batch, seq, hidden_dim)
    if num_heads > 1:
        shape = (batch, seq, num_heads, hidden_dim)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden_dim, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(batch * seq, max(1, rope_dim // 2), device="cuda", dtype=torch.float32)
    freqs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return x, weight, freqs


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
):
    output = x.float()
    if apply_rmsnorm:
        output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + eps)
    if apply_weight:
        output = output * weight.float()

    if rope_dim > 0:
        flat = output.reshape(-1, output.shape[-1])
        row_freqs = freqs.repeat_interleave(num_heads, dim=0)
        real = flat[:, -rope_dim::2].clone()
        imag = flat[:, -rope_dim + 1 :: 2].clone()
        cos = row_freqs[..., 0]
        sin = row_freqs[..., 1]
        if inverse_rope:
            sin = -sin
        flat[:, -rope_dim::2] = real * cos - imag * sin
        flat[:, -rope_dim + 1 :: 2] = imag * cos + real * sin
    return output.to(x.dtype)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "x_dtype",
        "weight_dtype",
        "freq_dtype",
        "odd_rope_dim",
        "unaligned_rope_pairs",
        "freq_rows",
        "x_layout",
    ],
)
def test_fused_dspark_rmsnorm_rope_support_gate_rejects_invalid_inputs(invalid_case):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        is_fused_dspark_rmsnorm_rope_supported,
    )

    inputs = list(_make_inputs(2, 5, 512, 64, 1))
    rope_dim = 64
    if invalid_case == "x_dtype":
        inputs[0] = inputs[0].float()
    elif invalid_case == "weight_dtype":
        inputs[1] = inputs[1].float()
    elif invalid_case == "freq_dtype":
        inputs[2] = inputs[2].to(torch.bfloat16)
    elif invalid_case == "odd_rope_dim":
        rope_dim = 63
    elif invalid_case == "unaligned_rope_pairs":
        rope_dim = 32
    elif invalid_case == "freq_rows":
        inputs[2] = inputs[2][:-1]
    else:
        inputs[0] = inputs[0].transpose(0, 1).contiguous().transpose(0, 1)

    assert not is_fused_dspark_rmsnorm_rope_supported(*inputs, num_heads=1, rope_dim=rope_dim)


def test_cute_dsl_dspark_rmsnorm_rope_rejects_invalid_inputs():
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
    )

    x, weight, freqs = _make_inputs(2, 5, 512, 64, 1)
    with pytest.raises(ValueError, match="requires contiguous BF16"):
        cute_dsl_dspark_rmsnorm_rope(x.float(), weight, freqs, 1, 64, 1e-6, True, True, False)


def test_dspark_rmsnorm_rope_kernel_rejects_invalid_num_heads():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dspark_rmsnorm_rope import (
        DSparkRMSNormRoPEKernel,
    )

    with pytest.raises(ValueError, match="num_heads must be positive"):
        DSparkRMSNormRoPEKernel(512, 64, 0, 1e-6, True, True, False)


@pytest.mark.parametrize(
    "hidden_dim,rope_dim,num_heads,apply_weight,apply_rmsnorm,inverse_rope",
    [
        (512, 64, 1, True, True, False),
        (512, 64, 1, True, False, False),
        (512, 64, 24, False, True, False),
        (512, 64, 24, False, False, True),
        (1024, 0, 1, True, True, False),
    ],
)
def test_fused_dspark_rmsnorm_rope_matches_reference(
    hidden_dim,
    rope_dim,
    num_heads,
    apply_weight,
    apply_rmsnorm,
    inverse_rope,
):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
    )

    inputs = _make_inputs(2, 5, hidden_dim, rope_dim, num_heads, seed=hidden_dim)
    eps = 1e-6
    actual = cute_dsl_dspark_rmsnorm_rope(
        *inputs,
        num_heads,
        rope_dim,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
    )
    expected = _reference(
        *inputs,
        num_heads,
        rope_dim,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_fused_dspark_rmsnorm_rope_cuda_graph_replay():
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
    )

    x, weight, freqs = _make_inputs(2, 5, 512, 64, 24, seed=1)
    args = (24, 64, 1e-6, False, True, False)
    cute_dsl_dspark_rmsnorm_rope(x, weight, freqs, *args)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cute_dsl_dspark_rmsnorm_rope(x, weight, freqs, *args)

    x.copy_(torch.randn_like(x))
    expected = _reference(x, weight, freqs, *args)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)


def test_fused_dspark_rmsnorm_rope_compiles_once_across_batches():
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        _compile_fused_dspark_rmsnorm_rope,
        cute_dsl_dspark_rmsnorm_rope,
    )

    _compile_fused_dspark_rmsnorm_rope.cache_clear()
    for batch in (1, 3):
        inputs = _make_inputs(batch, 5, 512, 64, 1, seed=batch)
        cute_dsl_dspark_rmsnorm_rope(*inputs, 1, 64, 1e-6, True, True, False)

    cache_info = _compile_fused_dspark_rmsnorm_rope.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1
