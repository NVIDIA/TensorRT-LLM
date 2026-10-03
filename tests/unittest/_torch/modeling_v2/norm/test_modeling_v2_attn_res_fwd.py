# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attn_res_fwd catalog entry.

A cell is (T, N): T tokens and N = K + 1 candidates, K snapshots plus the layer residual. Every cell
checks the four outputs' shapes and dtypes, bit-identical results from two identical calls, untouched
inputs, and closeness to an fp32 torch evaluation of the contract's Semantics under the metric of
main's op test (tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_op.py).

The op picks its kernel from (T, N): at T == 1 the single-CTA decode kernel for N in {1, 2, 4} and the
split-K cluster kernel for N in {8, 12}; the fixed-N = 12 online variant at (T, N) == (1024, 12); the
persistent online kernel everywhere else.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.attn_res_fwd import attn_res_fwd

assert torch.cuda.is_available(), "attn_res_fwd requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

HIDDEN_SIZE = 7168
RMS_EPS = 1e-6
# Kimi K3's 93 layers append a snapshot every 12 layers: at most 8 snapshots, so N runs 2 ... 9.
K3_NUM_CANDIDATES = tuple(range(2, 10))
DECODE_NUM_TOKENS = tuple(range(1, 9))
PREFILL_NUM_TOKENS = (300, 2048)
# The thresholds of main's op test.
MIN_COSINE = 0.999
MAX_RELATIVE_L2 = 3e-2
INPUT_NAMES = ("layer_residual", "block_residual", "res_weight", "rms_weight")
OUTPUT_NAMES = ("output", "rsigma", "probs", "logits")


def _make_inputs(num_tokens: int, num_candidates: int) -> tuple[torch.Tensor, ...]:
    """Inputs at the scales of main's op tests, seeded by the cell."""
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
    return layer_residual, block_residual, res_weight, rms_weight


def _reference(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 evaluation of the contract's Semantics; snapshots first, the layer residual last."""
    values = torch.cat((block_residual, layer_residual.unsqueeze(0)), dim=0).float()
    rsigma = torch.rsqrt(values.square().mean(dim=-1) + rms_eps)
    score_weight = rms_weight.float() * res_weight.float()
    logits = (values * rsigma.unsqueeze(-1) * score_weight).sum(dim=-1)
    probs = torch.softmax(logits, dim=0)
    output = (probs.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16)
    return output, rsigma, probs, logits


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers, so that equality means identical bits."""
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def _similarity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Cosine similarity and relative L2 error, the metric of main's op tests."""
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    return cosine, relative_l2


def _check(num_tokens: int, num_candidates: int, rms_eps: float = RMS_EPS) -> None:
    inputs = _make_inputs(num_tokens, num_candidates)
    before = [tensor.clone() for tensor in inputs]
    outputs = attn_res_fwd(*inputs, rms_eps)
    repeat = attn_res_fwd(*inputs, rms_eps)
    expected = _reference(*inputs, rms_eps)

    cell = f"T={num_tokens} N={num_candidates} rms_eps={rms_eps}"
    stats_shape = (num_candidates, num_tokens, 1)
    shapes = ((num_tokens, 1, HIDDEN_SIZE), stats_shape, stats_shape, stats_shape)
    dtypes = (torch.bfloat16, torch.float32, torch.float32, torch.float32)
    assert len(outputs) == len(OUTPUT_NAMES), f"{cell}: {len(outputs)} outputs"
    for name, actual, again, reference, shape, dtype in zip(
        OUTPUT_NAMES, outputs, repeat, expected, shapes, dtypes
    ):
        assert actual.shape == shape, f"{cell}: {name} shape {tuple(actual.shape)}"
        assert actual.dtype == dtype, f"{cell}: {name} dtype {actual.dtype}"
        assert actual.is_contiguous(), f"{cell}: {name} is not contiguous"
        assert actual.device == inputs[0].device, f"{cell}: {name} on {actual.device}"
        assert torch.equal(_bits(actual), _bits(again)), (
            f"{cell}: two identical calls disagree in {name}"
        )
        cosine, relative_l2 = _similarity(actual, reference)
        assert cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2, (
            f"{cell}: {name} cosine {cosine:.6f}, relative L2 {relative_l2:.3e}"
        )
    for name, old, new in zip(INPUT_NAMES, before, inputs):
        assert torch.equal(_bits(old), _bits(new)), f"{cell}: {name} was mutated"


def test_k3_decode_cells() -> None:
    """Kimi K3's N = 2 ... 9 at T = 1 ... 8: every T == 1 kernel, then the online kernel."""
    for num_tokens in DECODE_NUM_TOKENS:
        for num_candidates in K3_NUM_CANDIDATES:
            _check(num_tokens, num_candidates)


def test_k3_prefill_cells() -> None:
    """Kimi K3's N = 2 ... 9 at prefill token counts: the online kernel's CTAs loop over tokens."""
    for num_tokens in PREFILL_NUM_TOKENS:
        for num_candidates in K3_NUM_CANDIDATES:
            _check(num_tokens, num_candidates)


def test_other_dispatch_branches() -> None:
    """The kernels and N extremes outside Kimi K3's range, down to the fixed-N = 12 variant."""
    for num_tokens, num_candidates in (
        (1, 1),  # single-CTA kernel, no snapshot
        (1, 10),  # online kernel at T == 1
        (1, 11),  # online kernel at T == 1
        (1, 12),  # split-K kernel
        (8, 1),  # online kernel, a single candidate
        (8, 12),  # online kernel, three full chunks of four candidates
        (1024, 12),  # online kernel, fixed-N = 12 variant
    ):
        _check(num_tokens, num_candidates)


def test_rms_eps_enters_under_the_root() -> None:
    """rms_eps is added to the mean square inside the rsqrt; 1e-2 dominates the inputs' 2.5e-3."""
    for num_tokens, num_candidates in ((1, 2), (1, 8), (1, 9), (300, 9)):
        for rms_eps in (1e-5, 1e-2):
            _check(num_tokens, num_candidates, rms_eps)


def test_cuda_graph_replay_matches_eager() -> None:
    """One cell per kernel: a call captured in a CUDA graph replays the eager call's bits."""
    for num_tokens, num_candidates in ((1, 4), (1, 8), (4, 9)):
        inputs = _make_inputs(num_tokens, num_candidates)
        eager = attn_res_fwd(*inputs, RMS_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = attn_res_fwd(*inputs, RMS_EPS)
        # One replay before the poison: under cudaMallocAsync the captured outputs are graph allocations, backed by
        # memory only once the graph has run.
        graph.replay()
        for tensor in captured:
            tensor.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for name, expected, actual in zip(OUTPUT_NAMES, eager, captured):
            assert torch.equal(_bits(actual), _bits(expected)), (
                f"T={num_tokens} N={num_candidates}: replayed {name} differs from eager"
            )
        # Freed here rather than inside the next capture, where cudaMallocAsync's free would be part of it.
        del captured
