# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence of the fused greedy argmax+scatter and the tensor-op path."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.greedy_sample_kernels import (
    greedy_argmax_scatter,
    supports_greedy_argmax_scatter,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import fast_greedy_sample_kernel


def _reference(logits, new_tokens, dest_indices, beam_width):
    """The sequence the fused kernel replaces, run on a private copy."""
    out = new_tokens.clone()
    next_tokens = torch.argmax(logits, dim=-1).to(dtype=out.dtype)
    out.view(-1, *out.shape[2:]).scatter_(
        0,
        dest_indices.unsqueeze(1).expand(-1, beam_width),
        next_tokens.unsqueeze(1).expand(-1, beam_width),
    )
    return next_tokens, out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_rows,vocab_size,beam_width", [(1, 250_000, 1), (3, 4096, 2), (2, 97, 1)]
)
def test_matches_argmax_and_scatter(dtype, num_rows, vocab_size, beam_width):
    torch.manual_seed(num_rows * vocab_size + beam_width)
    num_slots = 8
    max_tokens = 2
    logits = torch.randn((num_rows, vocab_size), dtype=dtype, device="cuda")
    new_tokens = torch.full(
        (max_tokens, num_slots, beam_width), -7, dtype=torch.int32, device="cuda"
    )
    # Destinations out of order, so a kernel that assumed row == destination
    # would fail.
    dest = torch.tensor(
        [(num_slots * max_tokens - 1 - 2 * row) for row in range(num_rows)],
        dtype=torch.int64,
        device="cuda",
    )

    assert supports_greedy_argmax_scatter(logits, new_tokens)
    expected_tokens, expected_buffer = _reference(logits, new_tokens, dest, beam_width)
    tokens = greedy_argmax_scatter(logits, new_tokens, dest, beam_width)

    assert torch.equal(tokens, expected_tokens)
    assert torch.equal(new_tokens, expected_buffer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ties_take_the_lowest_index():
    """torch.argmax returns the first maximum; the packed key must agree."""
    vocab_size = 8192
    logits = torch.zeros((1, vocab_size), dtype=torch.float32, device="cuda")
    # Equal maxima spread across several reduction splits, including the last.
    for index in (5, 1000, 4096, vocab_size - 1):
        logits[0, index] = 3.5
    new_tokens = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
    dest = torch.zeros(1, dtype=torch.int64, device="cuda")

    tokens = greedy_argmax_scatter(logits, new_tokens, dest, 1)

    assert tokens.item() == 5
    assert torch.equal(tokens, torch.argmax(logits, dim=-1).to(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fill", [float("-inf"), -1e30, 0.0])
def test_degenerate_rows(fill):
    """A row with no distinguishable maximum still matches torch.argmax."""
    vocab_size = 3000
    logits = torch.full((1, vocab_size), fill, dtype=torch.float32, device="cuda")
    new_tokens = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
    dest = torch.zeros(1, dtype=torch.int64, device="cuda")

    tokens = greedy_argmax_scatter(logits, new_tokens, dest, 1)

    assert tokens.item() == torch.argmax(logits, dim=-1).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_negative_logits_order_correctly():
    """The float-to-integer key must keep negatives below every positive."""
    vocab_size = 5000
    logits = torch.full((1, vocab_size), -50.0, dtype=torch.float32, device="cuda")
    logits[0, 4321] = -0.5
    logits[0, 17] = -0.75
    new_tokens = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
    dest = torch.zeros(1, dtype=torch.int64, device="cuda")

    tokens = greedy_argmax_scatter(logits, new_tokens, dest, 1)

    assert tokens.item() == 4321


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fast_greedy_sample_kernel_dispatch_matches_reference():
    """The public entry point produces the same result on both of its paths."""
    torch.manual_seed(7)
    logits = torch.randn((2, 1024), dtype=torch.float32, device="cuda")
    dest = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
    baseline = torch.full((2, 4, 1), -1, dtype=torch.int32, device="cuda")
    expected_tokens, expected_buffer = _reference(logits, baseline, dest, 1)

    fused_buffer = baseline.clone()
    fused_tokens = fast_greedy_sample_kernel(logits, fused_buffer, dest, 1, None)
    assert torch.equal(fused_tokens, expected_tokens)
    assert torch.equal(fused_buffer, expected_buffer)

    # d2t present -> the unfused sequence, which must still agree when d2t is
    # all zeros.
    d2t_buffer = baseline.clone()
    d2t = torch.zeros(1024, dtype=torch.int32, device="cuda")
    d2t_tokens = fast_greedy_sample_kernel(logits, d2t_buffer, dest, 1, d2t)
    assert torch.equal(d2t_tokens, expected_tokens)
    assert torch.equal(d2t_buffer, expected_buffer)
