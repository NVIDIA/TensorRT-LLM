# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused routing: selection, bias semantics, tie policy, fallbacks and graph replay."""

from unittest.mock import patch

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.models import musespark_open_kernels
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenRoutingMethod,
    _routing_weights,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _router(bias):
    return MuseSparkOpenRoutingMethod(
        top_k=8, num_experts=256, callable_e_score_correction_bias=lambda: bias, renorm_eps=1e-6
    )


@cuda_only
@pytest.mark.parametrize("tokens", [1, 2, 16, 128])
@pytest.mark.parametrize("tied", [False, True])
def test_full_routing_selection_and_weights(tokens, tied):
    torch.manual_seed(910)
    logits = torch.randn(tokens, 256, device="cuda", dtype=torch.float32)
    bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.03
    if tied:
        logits.zero_()
        bias.zero_()
    router = _router(bias)
    selected, weights = router.apply(logits)
    scores = logits.sigmoid()
    if tied:
        # Exact ties go to the lowest expert ids.
        expected = torch.arange(8, device="cuda").expand(tokens, -1)
    else:
        expected = torch.topk(scores + bias, 8, sorted=True).indices
    torch.testing.assert_close(selected.long(), expected, rtol=0, atol=0)
    raw = scores.gather(1, expected)
    torch.testing.assert_close(
        weights, raw / (raw.sum(-1, keepdim=True) + 1e-6), rtol=1e-5, atol=1e-6
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = router.apply(logits)
    logits.normal_()
    want = router.apply(logits)
    graph.replay()
    torch.cuda.synchronize()
    for actual, reference in zip(got, want):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_full_routing_cpu_fallback():
    logits = torch.randn(2, 256)
    bias = torch.randn(256)
    router = _router(bias)
    ids, weights = router.apply(logits)
    scores = logits.sigmoid()
    expected = torch.topk(scores + bias, 8, sorted=False).indices
    torch.testing.assert_close(ids.long(), expected)
    raw = scores.gather(1, expected)
    torch.testing.assert_close(weights, raw / (raw.sum(-1, keepdim=True) + 1e-6))


@cuda_only
@pytest.mark.parametrize("tokens", [129, 512])
def test_batches_past_128_tokens_take_the_reference_selection(tokens):
    """Past 128 tokens selection is `torch.topk`; only the weights use a kernel."""
    torch.manual_seed(913)
    logits = torch.randn(tokens, 256, device="cuda")
    bias = torch.randn(256, device="cuda") * 0.03
    ids, weights = _router(bias).apply(logits)
    scores = logits.sigmoid()
    expected = torch.topk(scores + bias, 8, sorted=False).indices
    torch.testing.assert_close(ids.long(), expected, rtol=0, atol=0)
    raw = scores.gather(1, expected)
    torch.testing.assert_close(
        weights, raw / (raw.sum(-1, keepdim=True) + 1e-6), rtol=1e-6, atol=1e-7
    )


@cuda_only
@pytest.mark.parametrize("tokens", [1, 3, 256, 8192])
@pytest.mark.parametrize("case", ["random", "tiny", "zero", "ties"])
def test_the_routing_weights_kernel_matches_the_composition_exactly(tokens, case):
    """The weights kernel serves selections the fused routing does not make (over 128 tokens)."""
    torch.manual_seed(0)
    eps = 1e-15
    logits = torch.randn(tokens, 256, dtype=torch.float32, device="cuda")
    if case == "tiny":
        logits.fill_(-60.0)
    elif case == "ties":
        logits.zero_()
    scores = torch.sigmoid(logits)
    if case == "zero":
        scores.zero_()
    bias = torch.linspace(-0.2, 0.2, 256, device="cuda")
    if case == "ties":
        bias.zero_()
    _, indices = torch.topk(scores + bias, k=8, dim=-1, sorted=False)
    expected = scores.gather(1, indices)
    expected = expected / (expected.sum(-1, keepdim=True) + eps)

    kernel = musespark_open_kernels._routing_weights_kernel
    with patch.object(kernel, "run", wraps=kernel.run) as run:
        selected, weights = _routing_weights(scores, indices, eps, torch.float32)
    assert run.call_count == 1, "the composition answered instead of the kernel"

    assert selected.dtype == torch.int32 and weights.dtype == torch.float32
    assert selected.shape == weights.shape == (tokens, 8)
    torch.testing.assert_close(selected, indices.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)


@triton.jit
def _reference_argmax_routing_kernel(logits, bias, selected, weights, eps, EXPERTS: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.arange(0, EXPERTS)
    slots = tl.arange(0, 8)
    scores = tl.sigmoid(tl.load(logits + row * EXPERTS + expert))
    ranked = scores + tl.load(bias + expert)
    ids = tl.full((8,), 0, tl.int32)
    values = tl.full((8,), 0.0, tl.float32)
    for k in tl.static_range(8):
        best = tl.argmax(ranked, axis=0, tie_break_left=True)
        value = tl.sum(tl.where(expert == best, scores, 0.0), axis=0)
        ids = tl.where(slots == k, best, ids)
        values = tl.where(slots == k, value, values)
        ranked = tl.where(expert == best, -float("inf"), ranked)
    values = tl.div_rn(values, tl.sum(values, axis=0) + eps)
    tl.store(selected + row * 8 + slots, ids)
    tl.store(weights + row * 8 + slots, values)


@cuda_only
@pytest.mark.parametrize("tokens", [1, 2, 16, 128])
@torch.inference_mode()
def test_sorted_routing_matches_an_argmax_reference_on_mutable_inputs(tokens):
    """Repeated argmax with left tie-breaking states the selection order independently."""
    logits = torch.randn(tokens, 256, device="cuda")
    bias = torch.randn(256, device="cuda")
    router = _router(bias)
    router.apply(logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = router.apply(logits)
    for seed in range(32):
        torch.manual_seed(seed)
        logits.normal_()
        bias.normal_(0, 0.01)
        if seed == 0:
            logits.zero_()
            bias.fill_(-0.5)
        if seed == 1:
            logits.mul_(1e-6)
            bias.zero_()
        if seed == 2:
            bias.normal_(-1, 0.01)
        if seed == 3:
            logits.fill_(100)
            bias.zero_()
        if seed == 4:
            logits.floor_()
            bias.zero_()
        graph.replay()
        expected = (torch.empty_like(actual[0]), torch.empty_like(actual[1]))
        _reference_argmax_routing_kernel[(tokens,)](
            logits, bias, *expected, 1e-6, EXPERTS=256, num_warps=4
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-7)
