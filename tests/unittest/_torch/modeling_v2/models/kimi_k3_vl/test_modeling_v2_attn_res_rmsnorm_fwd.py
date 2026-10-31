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

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_rmsnorm_fwd import (
    attn_res_rmsnorm_fwd,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "attn_res_rmsnorm_fwd requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

HIDDEN_SIZE = 7168
RMS_EPS = 1e-6
# The thresholds of main's op test.
MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = 5e-3
INPUT_NAMES = ("layer_residual", "block_residual", "res_weight", "rms_weight", "output_rms_weight")


_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"


def _make_inputs(num_tokens: int, num_candidates: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Inputs at the scales of main's op test, seeded by the cell."""
    torch.manual_seed(seed)
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


def _check(cell) -> None:
    spec = cell.spec
    num_tokens, num_candidates = spec["tokens"], spec["candidates"]
    rms_eps, output_rms_eps = spec["eps"], spec["output_eps"]
    assert spec["hidden"] == HIDDEN_SIZE
    inputs = _make_inputs(num_tokens, num_candidates, zlib.crc32(cell.why.encode()) % 2**31)
    before = [tensor.clone() for tensor in inputs]
    with validating(attn_res_rmsnorm_fwd):
        output = attn_res_rmsnorm_fwd(*inputs, rms_eps, output_rms_eps)
        repeat = attn_res_rmsnorm_fwd(*inputs, rms_eps, output_rms_eps)
    expected = attn_res_rmsnorm_fwd.reference(*inputs, rms_eps, output_rms_eps)

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


@pytest.mark.parametrize(
    "cell", attn_res_rmsnorm_fwd.CELLS, ids=[c.why[:40] for c in attn_res_rmsnorm_fwd.CELLS]
)
def test_certified_cells(cell) -> None:
    _check(cell)


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


@pytest.mark.parametrize("num_tokens,num_candidates", ((1, 1), (2, 4), (8, 5), (32, 12)))
@pytest.mark.parametrize("early_trigger", (False, True))
def test_k3_fused_add_preserves_rounding_and_graph_inputs(
    num_tokens, num_candidates, early_trigger
):
    inputs = _make_inputs(num_tokens, num_candidates, seed=73)
    layer, block, res_weight, rms_weight, output_weight = inputs
    addend = layer.roll(1, dims=-1) * 0.25
    args = (layer, addend, block, res_weight, rms_weight, output_weight, RMS_EPS, RMS_EPS)
    updated, normalized = load_family_ops(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
    ).k3_attn_res_add_rmsnorm_fwd(*args, early_trigger=early_trigger)
    for replay in (False, True):
        if replay:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                updated, normalized = load_family_ops(
                    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
                ).k3_attn_res_add_rmsnorm_fwd(*args, early_trigger=early_trigger)
            layer.copy_(layer.roll(3, dims=-1))
            addend.neg_()
            graph.replay()
        expected_updated = layer + addend
        expected = attn_res_rmsnorm_fwd.reference(
            expected_updated, block, res_weight, rms_weight, output_weight, RMS_EPS, RMS_EPS
        )
        assert torch.equal(_bits(updated), _bits(expected_updated))
        cosine, relative_l2 = _similarity(normalized, expected)
        assert cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2
