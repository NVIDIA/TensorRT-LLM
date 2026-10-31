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

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_fwd import (
    attn_res_fwd,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "attn_res_fwd requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

HIDDEN_SIZE = 7168
RMS_EPS = 1e-6
# The thresholds of main's op test.
MIN_COSINE = 0.999
MAX_RELATIVE_L2 = 3e-2
INPUT_NAMES = ("layer_residual", "block_residual", "res_weight", "rms_weight")
OUTPUT_NAMES = ("output", "rsigma", "probs", "logits")


def _make_inputs(num_tokens: int, num_candidates: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Inputs at the scales of main's op tests, seeded by the cell."""
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
    return layer_residual, block_residual, res_weight, rms_weight


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


def _check(cell) -> None:
    spec = cell.spec
    num_tokens, num_candidates, rms_eps = spec["tokens"], spec["candidates"], spec["eps"]
    assert spec["hidden"] == HIDDEN_SIZE
    inputs = _make_inputs(num_tokens, num_candidates, zlib.crc32(cell.why.encode()) % 2**31)
    before = [tensor.clone() for tensor in inputs]
    with validating(attn_res_fwd):
        outputs = attn_res_fwd(*inputs, rms_eps)
        repeat = attn_res_fwd(*inputs, rms_eps)
    expected = attn_res_fwd.reference(*inputs, rms_eps)

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


@pytest.mark.parametrize("cell", attn_res_fwd.CELLS, ids=[c.why[:40] for c in attn_res_fwd.CELLS])
def test_certified_cells(cell) -> None:
    _check(cell)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("index", (1, 2, 3))
def test_guard_refuses_mixed_devices(index) -> None:
    inputs = list(_make_inputs(1, 2))
    other_device = (inputs[0].device.index + 1) % torch.cuda.device_count()
    inputs[index] = inputs[index].to(f"cuda:{other_device}")
    with pytest.raises(AssertionError, match="device"):
        with validating(attn_res_fwd):
            attn_res_fwd(*inputs, RMS_EPS)


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
