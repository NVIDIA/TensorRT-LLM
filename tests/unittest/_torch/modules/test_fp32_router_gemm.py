# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy and dispatch coverage for the FP32 router projection.

Neither kernel reproduces the cuBLAS result it stands in for, so the bar is not
bitwise equality. It is that both land closer to an FP64 reference than the TF32
path does, and that any shape neither kernel claims reaches `F.linear` bitwise.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fp32_router_gemm import (
    MAX_GEMV_TOKENS,
    cute_gate_ops,
    fp32_router_gemm,
    router_gemm,
    split_router_weight,
)

# MiniMax-M3's router: 128 experts over a 6144-wide hidden state.
M3_HIDDEN = 6144
M3_EXPERTS = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(num_tokens, hidden_size, num_experts, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float32, generator=gen)
    w = (
        torch.randn(num_experts, hidden_size, device="cuda", dtype=torch.float32, generator=gen)
        * 0.02
    )
    return x.to(dtype), w


def _max_err(actual, reference):
    return (actual.double() - reference).abs().max().item()


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16])
def test_matches_fp64_reference(num_tokens):
    x, w = _inputs(num_tokens, M3_HIDDEN, M3_EXPERTS)
    reference = x.double() @ w.double().t()

    out = fp32_router_gemm(x, w)

    assert out.shape == (num_tokens, M3_EXPERTS)
    assert out.dtype == torch.float32
    # Sum of 6144 products, so the FP32 accumulator carries a few ULP; the
    # reference row norm is order 1.
    assert _max_err(out, reference) < 2e-4


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_at_least_as_accurate_as_tf32_cublas(num_tokens):
    """The path being replaced rounds both operands to a 10-bit mantissa."""
    x, w = _inputs(num_tokens, M3_HIDDEN, M3_EXPERTS)
    reference = x.double() @ w.double().t()

    out = fp32_router_gemm(x, w)

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        cublas_tf32 = torch.nn.functional.linear(x.to(torch.float32), w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    assert _max_err(out, reference) <= _max_err(cublas_tf32, reference)


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_tracks_true_fp32_cublas(num_tokens):
    x, w = _inputs(num_tokens, M3_HIDDEN, M3_EXPERTS)

    out = fp32_router_gemm(x, w)

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        cublas_fp32 = torch.nn.functional.linear(x.to(torch.float32), w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    # Same precision, different summation order.
    torch.testing.assert_close(out, cublas_fp32, rtol=0, atol=2e-4)


@pytest.mark.parametrize("num_experts", [8, 33, 128, 256])
def test_expert_counts(num_experts):
    x, w = _inputs(4, 1024, num_experts)
    reference = x.double() @ w.double().t()

    assert _max_err(fp32_router_gemm(x, w), reference) < 2e-4


@pytest.mark.parametrize("hidden_size", [128, 1000, 6144])
def test_hidden_sizes_including_non_multiples(hidden_size):
    """The K loop masks its tail, so the hidden size need not divide the block."""
    x, w = _inputs(4, hidden_size, M3_EXPERTS)
    reference = x.double() @ w.double().t()

    assert _max_err(fp32_router_gemm(x, w), reference) < 2e-4


def test_prefill_shape_falls_back_bitwise():
    """Past the GEMV's window the call must land on `F.linear`, bitwise."""
    num_tokens = MAX_GEMV_TOKENS + 1
    x, w = _inputs(num_tokens, M3_HIDDEN, M3_EXPERTS)

    out = fp32_router_gemm(x, w)
    expected = torch.nn.functional.linear(x.to(torch.float32), w)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_non_contiguous_activation_falls_back():
    x, w = _inputs(4, M3_HIDDEN * 2, M3_EXPERTS)
    strided = x[:, ::2]
    weight = w[:, :M3_HIDDEN]
    assert strided.stride(1) != 1

    out = fp32_router_gemm(strided, weight)
    expected = torch.nn.functional.linear(strided.to(torch.float32), weight)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_fp32_activation_is_accepted():
    x, w = _inputs(4, M3_HIDDEN, M3_EXPERTS, dtype=torch.float32)
    reference = x.double() @ w.double().t()

    assert _max_err(fp32_router_gemm(x, w), reference) < 2e-4


def test_runs_under_cuda_graph():
    """Decode captures the step, so the launch has to be replayable."""
    x, w = _inputs(4, M3_HIDDEN, M3_EXPERTS)
    eager = fp32_router_gemm(x, w)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp32_router_gemm(x, w)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


# The dispatcher over both kernels. The CuTe DSL half is a soft dependency and
# wants Blackwell, so cases needing it skip elsewhere, under the same two
# conditions that leave the weight unsplit.

_gate_ops = cute_gate_ops()
needs_cute_gate = pytest.mark.skipif(
    _gate_ops is None or not _gate_ops.minimax_m3_gate_arch_is_supported(),
    reason="needs the CuTe DSL gate GEMM on Blackwell SM100",
)


@pytest.mark.parametrize("num_tokens", [1, 4, MAX_GEMV_TOKENS])
def test_dispatch_keeps_the_gemv_in_the_decode_band(num_tokens):
    """Up to the window the GEMV answers, split weight present or not."""
    x, w = _inputs(num_tokens, M3_HIDDEN, M3_EXPERTS)

    out = router_gemm(x, w, split_router_weight(w))

    torch.testing.assert_close(out, fp32_router_gemm(x, w), rtol=0, atol=0)


def test_dispatch_without_a_split_weight_falls_back_bitwise():
    """No CuTe DSL, or a gate that never loaded weights: `F.linear`, bitwise."""
    x, w = _inputs(MAX_GEMV_TOKENS + 1, M3_HIDDEN, M3_EXPERTS)

    out = router_gemm(x, w, None)
    expected = torch.nn.functional.linear(x.to(torch.float32), w)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@needs_cute_gate
def test_split_weight_reconstructs_the_fp32_weight():
    """The terms are a rewriting of the weight, not an approximation of it."""
    _, w = _inputs(1, M3_HIDDEN, M3_EXPERTS)
    terms = cute_gate_ops().MINIMAX_M3_GATE_GEMM_TERMS

    w_split = split_router_weight(w)

    assert w_split.shape == (terms * M3_EXPERTS, M3_HIDDEN)
    assert w_split.dtype == torch.bfloat16
    summed = w_split.view(terms, M3_EXPERTS, M3_HIDDEN).to(torch.float32).sum(0)
    # Two bf16 terms carry ~16 mantissa bits of the original.
    torch.testing.assert_close(summed, w, rtol=1e-4, atol=0)


@needs_cute_gate
@pytest.mark.parametrize("num_tokens", [MAX_GEMV_TOKENS + 1, 64, 1024])
def test_dispatch_above_the_band_beats_tf32_cublas(num_tokens):
    """The wide path exists to be more accurate than the TF32 cuBLAS pair."""
    x, w = _inputs(num_tokens, M3_HIDDEN, M3_EXPERTS)
    reference = x.double() @ w.double().t()

    out = router_gemm(x, w, split_router_weight(w))

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        cublas_tf32 = torch.nn.functional.linear(x.to(torch.float32), w)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    assert out.shape == (num_tokens, M3_EXPERTS)
    assert out.dtype == torch.float32
    assert _max_err(out, reference) <= _max_err(cublas_tf32, reference)
