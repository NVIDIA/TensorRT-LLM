# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_fwd entry.

A cell is (T, N): T tokens and N = K + 1 candidates, K snapshots plus the layer residual. Every cell
checks the four outputs' shapes and dtypes, bit-identical results from two identical calls, untouched
inputs, and closeness to the entry's fp32 reference.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_fwd import (
    attn_res_fwd,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 and sm_107",
)

HIDDEN_SIZE = 7168
RMS_EPS = 1e-5
# Measured on GB300 over every cell: the mixture within relative L2 7.5e-5 of the fp32 reference (at
# most 1.7 bf16 ulp of a row's largest magnitude per element: one bf16 rounding on each side plus
# fast-math softmax), the fp32 statistics within 2.5e-7; cosine >= 0.9999998 everywhere. The bands
# leave more than an order of magnitude; a dropped or doubled candidate moves the mixture by tens of
# percent.
MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = (1e-3, 1e-5, 1e-5, 1e-5)
INPUT_NAMES = ("layer_residual", "block_residual", "res_weight", "rms_weight")
OUTPUT_NAMES = ("output", "rsigma", "probs", "logits")


def _make_inputs(num_tokens: int, num_candidates: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Residual-stream scale values, with a scoring weight that spreads the softmax over candidates."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    layer_residual = (randn(num_tokens, 1, HIDDEN_SIZE) * 0.05).to(torch.bfloat16)
    block_residual = (randn(num_candidates - 1, num_tokens, 1, HIDDEN_SIZE) * 0.05).to(
        torch.bfloat16
    )
    res_weight = (randn(HIDDEN_SIZE) * 0.02).to(torch.bfloat16)
    rms_weight = (1 + randn(HIDDEN_SIZE) * 0.02).to(torch.bfloat16)
    return layer_residual, block_residual, res_weight, rms_weight


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def _similarity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    a, e = actual.float().flatten(), expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    relative_l2 = ((a - e).norm() / (e.norm() + 1e-12)).item()
    return cosine, relative_l2


@pytest.mark.parametrize("cell", attn_res_fwd.CELLS, ids=[c.why[:48] for c in attn_res_fwd.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(attn_res_fwd)
    spec = cell.spec
    num_tokens, num_candidates, rms_eps = spec["tokens"], spec["candidates"], spec["eps"]
    assert spec["hidden"] == HIDDEN_SIZE
    inputs = _make_inputs(num_tokens, num_candidates, zlib.crc32(cell.why.encode()) % 2**31)
    before = [tensor.clone() for tensor in inputs]
    outputs = attn_res_fwd(*inputs, rms_eps)
    repeat = attn_res_fwd(*inputs, rms_eps)
    expected = attn_res_fwd.reference(*inputs, rms_eps)

    stats_shape = (num_candidates, num_tokens, 1)
    shapes = ((num_tokens, 1, HIDDEN_SIZE), stats_shape, stats_shape, stats_shape)
    dtypes = (torch.bfloat16, torch.float32, torch.float32, torch.float32)
    for name, actual, again, ref, shape, dtype, max_l2 in zip(
        OUTPUT_NAMES, outputs, repeat, expected, shapes, dtypes, MAX_RELATIVE_L2
    ):
        assert actual.shape == shape and actual.dtype == dtype, (name, actual.shape, actual.dtype)
        assert torch.equal(_bits(actual), _bits(again)), f"two identical calls disagree in {name}"
        cosine, relative_l2 = _similarity(actual, ref)
        assert cosine > MIN_COSINE and relative_l2 < max_l2, (
            f"{name}: cosine {cosine:.7f}, relative L2 {relative_l2:.3e}"
        )
    for name, old, new in zip(INPUT_NAMES, before, inputs):
        assert torch.equal(_bits(old), _bits(new)), f"{name} was mutated"


def test_gate_refuses_a_dropped_candidate() -> None:
    """The band must refuse a mixture that leaves one snapshot out."""
    inputs = _make_inputs(8, 5, seed=3)
    out = attn_res_fwd(*inputs, RMS_EPS)[0]
    layer_residual, block_residual, res_weight, rms_weight = inputs
    wrong = attn_res_fwd.reference(
        layer_residual, block_residual[1:], res_weight, rms_weight, RMS_EPS
    )
    cosine, relative_l2 = _similarity(out, wrong[0])
    assert not (cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2[0])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("index", (1, 2, 3))
def test_guard_refuses_mixed_devices(index) -> None:
    inputs = list(_make_inputs(1, 2))
    other = (inputs[0].device.index + 1) % torch.cuda.device_count()
    inputs[index] = inputs[index].to(f"cuda:{other}")
    with pytest.raises(AssertionError, match="device"):
        with validating(attn_res_fwd):
            attn_res_fwd(*inputs, RMS_EPS)


def test_cuda_graph_replay_matches_eager() -> None:
    """One cell per kernel: a call captured in a CUDA graph replays the eager call's bits."""
    for num_tokens, num_candidates in ((1, 4), (1, 8), (5, 9)):
        inputs = _make_inputs(num_tokens, num_candidates)
        eager = attn_res_fwd(*inputs, RMS_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = attn_res_fwd(*inputs, RMS_EPS)
        graph.replay()
        for tensor in captured:
            tensor.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for name, expected, actual in zip(OUTPUT_NAMES, eager, captured):
            assert torch.equal(_bits(actual), _bits(expected)), (
                f"T={num_tokens} N={num_candidates}: replayed {name} differs from eager"
            )
        del captured
