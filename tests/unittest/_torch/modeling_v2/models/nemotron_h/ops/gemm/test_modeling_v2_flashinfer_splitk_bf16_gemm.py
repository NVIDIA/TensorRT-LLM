# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_splitk_bf16_gemm family op."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.gemm.flashinfer_splitk_bf16_gemm import (  # noqa: E501
    flashinfer_splitk_bf16_gemm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402 — needs the path declared above
    ULP,
    assert_within_ulp,
    validating,
)

assert torch.cuda.is_available(), "flashinfer_splitk_bf16_gemm requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference: `cublas_mm`'s band (4 ulp of the row
    scale element-wise, 2 ulp of relative RMS). Kernel and reference consume bit-identical
    operands and differ only in the order fp32 partial sums are added."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(m: int, k: int, n: int, seed: int):
    """`mat_b` is built as [N, K] and transposed: the op reads it column-major, which is what a
    stored weight's `.t()` already is."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    mat_b = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16).t()
    return mat_a, mat_b


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
@pytest.mark.parametrize("pdl", [False, True])
def test_certified_cells(cell, pdl) -> None:
    spec = cell.spec
    mat_a, mat_b = _build(spec["m"], spec["k"], spec["n"], zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(mat_a, mat_b, pdl)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype is torch.bfloat16
    _compare(out, op.reference(mat_a, mat_b, pdl))


@pytest.mark.parametrize("m", [3, 24])
def test_row_counts_between_cells_take_the_next_cell_tactic(m) -> None:
    """A captured batch the CUDA-graph padding lands between two measured cells (24 rows, say)
    runs the next larger cell's tactic, and is as exact."""
    mat_a, mat_b = _build(m, 8192, 35072, seed=21 + m)
    with validating(op):
        out = op(mat_a, mat_b, True)
    _compare(out, op.reference(mat_a, mat_b, True))


@pytest.mark.parametrize("m", [4, 16])
def test_repeated_calls_are_bitwise_identical(m) -> None:
    """The result depends only on the shape and the fixed tactic, so ranks that run the same
    replicated projection get the same bits."""
    mat_a, mat_b = _build(m, 16384, 8192, seed=11)
    first = op(mat_a, mat_b, True)
    for _ in range(3):
        assert torch.equal(op(mat_a, mat_b, True), first)


def test_replays_under_a_cuda_graph() -> None:
    """Captured once, replayed on new activations: each replay is bitwise the eager call."""
    mat_a, mat_b = _build(16, 8192, 10240, seed=12)
    op(mat_a, mat_b, True)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(mat_a, mat_b, True)
    for seed in (13, 14):
        fresh, _ = _build(16, 8192, 10240, seed=seed)
        mat_a.copy_(fresh)
        graph.replay()
        assert torch.equal(captured, op(mat_a, mat_b, True))
