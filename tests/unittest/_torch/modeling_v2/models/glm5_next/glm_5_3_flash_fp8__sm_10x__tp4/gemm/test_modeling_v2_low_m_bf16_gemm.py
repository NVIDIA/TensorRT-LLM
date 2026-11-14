# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 low_m_bf16_gemm entry.

Every row count a table entry runs compiles its own CuTe DSL kernel, seconds each. The default
run drives the row counts at the tactic boundaries of each shape; ``LOW_M_BF16_GEMM_TEST_ALL_M=1``
drives every row count from 1 to 32.
"""

import os
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.gemm.low_m_bf16_gemm import (  # noqa: E501
    _CHOICES,
    low_m_choice,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.gemm.low_m_bf16_gemm import (  # noqa: E501
    low_m_bf16_gemm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Kernel and reference multiply the same BF16 operands exactly in FP32 and differ only in
    the order of the FP32 sums (split-K slices, the direct kernel's per-thread partials, cuBLAS),
    about 1e-6 of a row's scale: at most rare one-ulp flips of the single BF16 rounding. 4 ulp
    element-wise and 2 ulp RMS admit those; a lost K slice or wrong tile is off by hundreds."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _build(m: int, n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    return x, torch.nn.Parameter(weight, requires_grad=False)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    x, weight = _build(s["m"], s["n"], s["k"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    _compare(op(x, weight), op.reference(x, weight.detach()))


#: The largest row count a table entry serves.
_MAX_M = 32

_ALL_M = os.environ.get("LOW_M_BF16_GEMM_TEST_ALL_M", "0") == "1"


def _boundary_row_counts(n: int, k: int) -> list[int]:
    """The first and last row count of every run of row counts that take the same choice
    (kernel and tactic, or cuBLAS), up to ``_MAX_M``: each tactic at its smallest and largest
    M, both sides of every switch, and a direct tactic's row tile skipped where it does not
    divide M."""
    choices = [low_m_choice(m, n, k) for m in range(1, _MAX_M + 1)]
    ends = {1, _MAX_M}
    for m in range(2, _MAX_M + 1):
        if choices[m - 1] != choices[m - 2]:
            ends.update((m - 1, m))
    return sorted(ends)


@pytest.mark.parametrize("shape", sorted(_CHOICES), ids=lambda s: f"{s[0]}x{s[1]}")
def test_row_counts(shape: tuple[int, int]) -> None:
    """Every row count driven picks a tactic its kernel accepts (direct tactics need their row
    tile to divide M) and computes the product: the tactic boundaries, or with
    ``LOW_M_BF16_GEMM_TEST_ALL_M=1`` every row count of the table's range."""
    n, k = shape
    row_counts = range(1, _MAX_M + 1) if _ALL_M else _boundary_row_counts(n, k)
    for m in row_counts:
        x, weight = _build(m, n, k, seed=m)
        choice = low_m_choice(m, n, k)
        if choice is not None and choice[0] == "direct":
            rows_per_block = choice[1][2]
            assert rows_per_block <= m and m % rows_per_block == 0
        _compare(op(x, weight), op.reference(x, weight.detach()))


def test_cuda_graph_capture_and_replay() -> None:
    """The decode step is captured: the op must replay on new contents of the same input."""
    s = op.CELLS[1].spec
    x, weight = _build(s["m"], s["n"], s["k"], seed=31)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, weight)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, weight)
    fresh, _ = _build(s["m"], s["n"], s["k"], seed=32)
    x.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    _compare(out, op.reference(x, weight.detach()))
