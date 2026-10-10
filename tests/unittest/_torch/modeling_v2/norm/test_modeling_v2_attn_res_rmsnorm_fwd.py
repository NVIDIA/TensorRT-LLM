# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attn_res_rmsnorm_fwd catalog entry.

A cell is (T, N): T tokens and N = K + 1 candidates, K snapshots plus the layer residual. Every cell
checks the output's shape and dtype, bit-identical results from two identical calls, untouched inputs,
and closeness to an fp32 torch evaluation of the contract's Semantics, with both bf16 rounding
boundaries, under the metric of main's op test
(tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_rmsnorm_op.py).

The kernel depends on N only: one CTA per token for N <= 4, one cluster of 8 CTAs per token for
N >= 5. With PDL enabled both release their dependents right after their own grid-dependency wait.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.attn_res_rmsnorm_fwd import (
    attn_res_rmsnorm_fwd,
)

assert torch.cuda.is_available(), "attn_res_rmsnorm_fwd requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

HIDDEN_SIZE = 7168
RMS_EPS = 1e-6
# Kimi K3's 93 layers append a snapshot every 12 layers: at most 8 snapshots, so N runs 2 ... 9.
K3_NUM_CANDIDATES = tuple(range(2, 10))
DECODE_NUM_TOKENS = tuple(range(1, 9))
# 32 is the fused path's token ceiling when KIMI_K3_ATTN_RES_TOPOLOGY is on.
LARGER_NUM_TOKENS = (32, 300)
# The thresholds of main's op test.
MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = 5e-3
INPUT_NAMES = ("layer_residual", "block_residual", "res_weight", "rms_weight", "output_rms_weight")


def _make_inputs(num_tokens: int, num_candidates: int) -> tuple[torch.Tensor, ...]:
    """Inputs at the scales of main's op test, seeded by the cell."""
    torch.manual_seed(97 * num_tokens + num_candidates)
    layer_residual = (
        torch.randn(num_tokens, 1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    )
    block_residual = (
        torch.randn(
            num_candidates - 1, num_tokens, 1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        * 0.05
    )
    res_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    rms_weight = 1 + torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    output_rms_weight = 1 + torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02
    return layer_residual, block_residual, res_weight, rms_weight, output_rms_weight


def _reference(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
) -> torch.Tensor:
    """fp32 evaluation of the contract's Semantics, rounding where the kernel rounds."""
    values = torch.cat((block_residual, layer_residual.unsqueeze(0)), dim=0).float()
    rsigma = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
    logits = (values * rsigma * (rms_weight.float() * res_weight.float())).sum(dim=-1)
    probs = torch.softmax(logits, dim=0)
    mixed = (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16).float()
    normed = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + output_rms_eps)
    # bf16 times bf16: the exact product, rounded to bf16 once more.
    return output_rms_weight * normed.to(torch.bfloat16)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers, so that equality means identical bits."""
    return tensor.view(torch.int16)


def _similarity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Cosine similarity and relative L2 error, the metric of main's op tests."""
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    return cosine, relative_l2


def _check(
    num_tokens: int,
    num_candidates: int,
    rms_eps: float = RMS_EPS,
    output_rms_eps: float = RMS_EPS,
) -> None:
    inputs = _make_inputs(num_tokens, num_candidates)
    before = [tensor.clone() for tensor in inputs]
    output = attn_res_rmsnorm_fwd(*inputs, rms_eps, output_rms_eps)
    repeat = attn_res_rmsnorm_fwd(*inputs, rms_eps, output_rms_eps)
    expected = _reference(*inputs, rms_eps, output_rms_eps)

    cell = f"T={num_tokens} N={num_candidates} eps=({rms_eps}, {output_rms_eps})"
    assert output.shape == (num_tokens, 1, HIDDEN_SIZE), f"{cell}: shape {tuple(output.shape)}"
    assert output.dtype == torch.bfloat16, f"{cell}: dtype {output.dtype}"
    assert output.is_contiguous(), f"{cell}: output is not contiguous"
    assert output.device == inputs[0].device, f"{cell}: output on {output.device}"
    assert torch.equal(_bits(output), _bits(repeat)), f"{cell}: two identical calls disagree"
    cosine, relative_l2 = _similarity(output, expected)
    assert cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2, (
        f"{cell}: cosine {cosine:.6f}, relative L2 {relative_l2:.3e}"
    )
    for name, old, new in zip(INPUT_NAMES, before, inputs):
        assert torch.equal(_bits(old), _bits(new)), f"{cell}: {name} was mutated"


def test_k3_decode_cells() -> None:
    """Kimi K3's N = 2 ... 9 at T = 1 ... 8, on both kernels."""
    for num_tokens in DECODE_NUM_TOKENS:
        for num_candidates in K3_NUM_CANDIDATES:
            _check(num_tokens, num_candidates)


def test_larger_token_counts() -> None:
    """Kimi K3's N = 2 ... 9 at more tokens: more CTAs or clusters, the same per-token kernel."""
    for num_tokens in LARGER_NUM_TOKENS:
        for num_candidates in K3_NUM_CANDIDATES:
            _check(num_tokens, num_candidates)


def test_other_candidate_counts() -> None:
    """N = 1 (single-CTA kernel, no snapshot) and N = 10 ... 12 (split-K), outside Kimi K3's range."""
    for num_tokens in (1, 8):
        for num_candidates in (1, 10, 11, 12):
            _check(num_tokens, num_candidates)


def test_each_eps_feeds_its_own_norm() -> None:
    """rms_eps scores the candidates and output_rms_eps scales the output; 1e-2 dominates either."""
    for num_tokens, num_candidates in ((1, 4), (1, 9), (8, 2)):
        for rms_eps, output_rms_eps in ((1e-5, 1e-5), (1e-2, 1e-6), (1e-6, 1e-2)):
            _check(num_tokens, num_candidates, rms_eps, output_rms_eps)


def test_cuda_graph_replay_matches_eager() -> None:
    """One cell per kernel: a call captured in a CUDA graph replays the eager call's bits."""
    for num_tokens, num_candidates in ((1, 4), (1, 9), (8, 5)):
        inputs = _make_inputs(num_tokens, num_candidates)
        eager = attn_res_rmsnorm_fwd(*inputs, RMS_EPS, RMS_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = attn_res_rmsnorm_fwd(*inputs, RMS_EPS, RMS_EPS)
        # One replay before the poison: under cudaMallocAsync the captured output is a graph allocation, backed by
        # memory only once the graph has run.
        graph.replay()
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(captured), _bits(eager)), (
            f"T={num_tokens} N={num_candidates}: replay differs from eager"
        )
        # Freed here rather than inside the next capture, where cudaMallocAsync's free would be part of it.
        del captured


def test_chained_calls_wait_for_their_input() -> None:
    """A call reading the previous call's output reads it settled, although that call triggers early.

    With PDL enabled the first call releases its dependents before it writes its output, so the second
    call can start while the first still runs; its grid-dependency wait must hold its reads back.
    """
    for num_tokens, num_candidates in ((1, 4), (1, 9), (8, 4), (8, 9)):
        layer_residual, block_residual, res_weight, rms_weight, output_rms_weight = _make_inputs(
            num_tokens, num_candidates
        )
        weights = (res_weight, rms_weight, output_rms_weight, RMS_EPS, RMS_EPS)
        first = attn_res_rmsnorm_fwd(layer_residual, block_residual, *weights)
        chained = attn_res_rmsnorm_fwd(first, block_residual, *weights)
        torch.cuda.synchronize()
        settled = attn_res_rmsnorm_fwd(first.clone(), block_residual, *weights)
        assert torch.equal(_bits(chained), _bits(settled)), (
            f"T={num_tokens} N={num_candidates}: the chained call read an unsettled input"
        )
