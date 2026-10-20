# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_direct_bf16_gemm catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.flashinfer_direct_bf16_gemm import (  # noqa: E501
    flashinfer_direct_bf16_gemm as op,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402 — needs the path declared above
    ULP,
    assert_within_ulp,
    validating,
)

assert torch.cuda.is_available(), "flashinfer_direct_bf16_gemm requires a CUDA device"


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


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
@pytest.mark.parametrize("pdl", [False, True])
def test_certified_cells(cell, pdl) -> None:
    spec = cell.spec
    mat_a, mat_b = _build(spec["m"], spec["k"], spec["n"], zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(mat_a, mat_b, pdl)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype is torch.bfloat16
    _compare(out, op.reference(mat_a, mat_b, pdl))


@pytest.mark.parametrize("m", [1, 2])
def test_repeated_calls_are_bitwise_identical(m) -> None:
    """The result depends only on the shape and the fixed tactic, so ranks that run the same
    replicated projection get the same bits."""
    mat_a, mat_b = _build(m, 8192, 1280, seed=11)
    first = op(mat_a, mat_b, True)
    for _ in range(3):
        assert torch.equal(op(mat_a, mat_b, True), first)


def test_replays_under_a_cuda_graph() -> None:
    """Captured once, replayed on new activations: each replay is bitwise the eager call."""
    mat_a, mat_b = _build(1, 8192, 4384, seed=12)
    op(mat_a, mat_b, True)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(mat_a, mat_b, True)
    for seed in (13, 14):
        fresh, _ = _build(1, 8192, 4384, seed=seed)
        mat_a.copy_(fresh)
        graph.replay()
        assert torch.equal(captured, op(mat_a, mat_b, True))


@pytest.mark.parametrize("case", ["row_major_mat_b", "strided_mat_a", "m_33"])
def test_kernel_refuses_loudly_what_is_valid_does_not_restate(case) -> None:
    mat_a, mat_b = _build(2, 8192, 1280, seed=15)
    if case == "row_major_mat_b":
        mat_b = mat_b.contiguous()
    elif case == "strided_mat_a":
        mat_a = torch.randn(2, 2 * 8192, device="cuda").to(torch.bfloat16)[:, ::2]
    elif case == "m_33":
        mat_a, mat_b = _build(33, 8192, 1280, seed=15)
    with pytest.raises(ValueError):
        with validating(op):
            op(mat_a, mat_b, True)
