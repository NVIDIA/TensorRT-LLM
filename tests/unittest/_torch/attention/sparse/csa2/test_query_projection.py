# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 MXFP8 query projection with GPT-J RoPE and unit-scale E4M3 output."""

import pytest
import torch

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="The CuTe DSL query projection requires SM100/SM103",
)


def _problem(m, heads, k=256, *, seed=0, positions_dtype=torch.int32, nan_padding=False):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8

    gen = torch.Generator(device="cuda").manual_seed(seed)
    latent = (torch.randn(m, k, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    gamma = (torch.rand(k, device="cuda", generator=gen) + 0.5).to(torch.bfloat16)
    _, activation, activation_scale = rms_norm_mxfp8(latent, gamma, 1e-6)
    if nan_padding:
        # NaN UE8M0 scales on the padded rows turn those GEMM rows into NaN.
        rows = (m + 127) // 128 * 128
        linear = torch.full((rows, k // 32), 255, device="cuda", dtype=torch.uint8)
        linear[:m] = torch.ops.trtllm.block_scale_interleave_reverse(
            activation_scale.view(rows, k // 32)
        )[:m]
        activation_scale = torch.ops.trtllm.block_scale_interleave(linear).reshape(-1)
    n = heads * 512
    weight = (torch.randn(n, k, device="cuda", generator=gen) * 0.6).to(torch.float8_e4m3fn)
    weight_scale = torch.ops.trtllm.block_scale_interleave(
        torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8, generator=gen)
    ).reshape(-1)
    positions = torch.randint(0, 64, (m,), device="cuda", generator=gen).to(positions_dtype)
    angle = torch.rand(64, 32, device="cuda", generator=gen) * 6.3
    cos_sin = torch.stack((angle.cos(), angle.sin()), dim=1).contiguous()
    alpha = torch.ones(1, device="cuda")
    return activation, activation_scale, weight, weight_scale, alpha, positions, cos_sin


def _dequantized_gemm(activation, activation_scale, weight, weight_scale):
    m, k = activation.shape
    n = weight.shape[0]
    rows = (m + 127) // 128 * 128

    def dequant(values, scale, count, padded):
        linear = torch.ops.trtllm.block_scale_interleave_reverse(scale.view(padded, k // 32))
        factor = torch.exp2(linear[:count].float() - 127).repeat_interleave(32, dim=1)
        return values.float() * factor

    return dequant(activation, activation_scale, m, rows) @ dequant(weight, weight_scale, n, n).T


@skip_pre_blackwell
@pytest.mark.parametrize("m,heads", [(7, 8), (129, 16), (300, 8)])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_query_gemm_rope_matches_cuda_rope(m, heads, positions_dtype):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.query_projection import query_gemm_rope

    inputs = _problem(m, heads, seed=m, positions_dtype=positions_dtype)
    q, quant_q = query_gemm_rope(*inputs, quantize=True)
    activation, activation_scale, weight, weight_scale, _, positions, cos_sin = inputs
    reference = (
        _dequantized_gemm(activation, activation_scale, weight, weight_scale)
        .to(torch.bfloat16)
        .view(m, heads, 512)
    )
    torch.ops.trtllm.mla_rope_inplace(
        reference, positions.int(), cos_sin, heads, 448, 64, False, False
    )
    torch.testing.assert_close(q.view(m, heads, 512), reference, atol=0.02, rtol=0.02)
    unit, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        q, torch.ones(1, device="cuda")
    )
    assert torch.equal(quant_q.view(torch.uint8), unit.view(torch.uint8))
    # The epilogue rotates its BF16 values exactly as the CUDA RoPE kernel would.
    identity = torch.zeros_like(cos_sin)
    identity[:, 0] = 1.0
    unrotated, _ = query_gemm_rope(*inputs[:6], identity, quantize=False)
    expected = unrotated.view(m, heads, 512).clone()
    torch.ops.trtllm.mla_rope_inplace(
        expected, positions.int(), cos_sin, heads, 448, 64, False, False
    )
    assert torch.equal(expected.view(torch.int16), q.view(m, heads, 512).view(torch.int16))


@skip_pre_blackwell
@pytest.mark.parametrize("m,heads", [(7, 8), (129, 16), (300, 8)])
@pytest.mark.parametrize("nan_padding", [False, True])
@torch.inference_mode()
def test_fp8_query_gemm_matches_query_gemm_rope(m, heads, nan_padding):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.query_projection import (
        fp8_query_gemm,
        query_gemm_rope,
    )

    inputs = _problem(m, heads, seed=m + 1, nan_padding=nan_padding)
    q, expected = query_gemm_rope(*inputs, quantize=True)
    quant_q = torch.full_like(expected, float("nan"))
    word = torch.zeros(1, dtype=torch.int32, device="cuda")
    assert fp8_query_gemm(*inputs, quant_q, word)
    assert torch.equal(quant_q.view(torch.uint8), expected.view(torch.uint8))
    largest = q.float().abs().max()
    assert torch.isfinite(largest)
    assert word.item() == largest.view(torch.int32).item()
    # The word only ever rises: a larger earlier value survives.
    word.fill_((largest * 2).view(torch.int32).item())
    assert fp8_query_gemm(*inputs, quant_q, word)
    assert word.item() == (largest * 2).view(torch.int32).item()


@skip_pre_blackwell
@torch.inference_mode()
def test_fp8_query_gemm_graph_replay():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.query_projection import (
        fp8_query_gemm,
        query_gemm_rope,
    )

    activation, activation_scale, weight, weight_scale, alpha, positions, cos_sin = _problem(
        200, 8, seed=3
    )
    quant_q = torch.empty(200, 8 * 512, device="cuda", dtype=torch.float8_e4m3fn)
    word = torch.zeros(1, dtype=torch.int32, device="cuda")
    arguments = (activation, activation_scale, weight, weight_scale, alpha, positions, cos_sin)
    assert fp8_query_gemm(*arguments, quant_q, word)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert fp8_query_gemm(*arguments, quant_q, word)
    for step in range(2):
        word.zero_()
        positions.copy_(torch.randperm(64, device="cuda")[:1].expand(200) + step)
        weight.copy_(torch.flip(weight.view(torch.uint8), (1,)).view(torch.float8_e4m3fn))
        graph.replay()
        q, expected = query_gemm_rope(*arguments, quantize=True)
        assert torch.equal(quant_q.view(torch.uint8), expected.view(torch.uint8))
        assert word.item() == q.float().abs().max().view(torch.int32).item()


@skip_pre_blackwell
@torch.inference_mode()
def test_fp8_query_gemm_does_not_compile_during_capture():
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import query_projection

    inputs = _problem(16, 8, seed=4)
    quant_q = torch.empty(16, 8 * 512, device="cuda", dtype=torch.float8_e4m3fn)
    word = torch.zeros(1, dtype=torch.int32, device="cuda")
    saved = dict(query_projection._COMPILED)
    query_projection._COMPILED.clear()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert not query_projection.fp8_query_gemm(*inputs, quant_q, word)
    finally:
        query_projection._COMPILED.clear()
        query_projection._COMPILED.update(saved)


@skip_pre_blackwell
@pytest.mark.parametrize("exponent", [0, 3, 6])
@torch.inference_mode()
def test_rescale_fp8_query_matches_direct_quantization(exponent):
    """Normal unit-scale codes rescale exactly; only unit-scale subnormals keep their grid."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.query_projection import (
        rescale_fp8_query,
    )

    gen = torch.Generator(device="cuda").manual_seed(exponent)
    q = torch.randn(37, 4096, device="cuda", generator=gen)
    q = (q * torch.exp2(torch.randint(-9, 3, q.shape, device="cuda", generator=gen))).to(
        torch.bfloat16
    )
    q[0, :8] = torch.tensor(
        [0.0, -0.0, 448.0 / 2**exponent, -1.0, 2.0**-6, 2.0**-7, 2.0**-10, 1e-30]
    )
    scale = torch.full((1,), 2.0**-exponent, device="cuda")
    expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, scale)
    unit, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        q, torch.ones(1, device="cuda")
    )
    original = unit.clone()
    rescale_fp8_query(unit, scale)
    if exponent == 0:
        assert torch.equal(unit.view(torch.uint8), original.view(torch.uint8))
    normal = q.float().abs() >= 2.0**-6
    assert torch.equal(unit.view(torch.uint8)[normal], expected.view(torch.uint8)[normal])
    # A unit-scale subnormal is off by at most half its own 2**-9 step, scaled.
    error = (unit.float() - q.float() * 2.0**exponent).abs()
    assert bool((error[~normal] <= 2.0 ** (exponent - 10)).all())
