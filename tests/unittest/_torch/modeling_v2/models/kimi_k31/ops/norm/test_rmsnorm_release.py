# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 q_a RMSNorm with the launch-time PDL release: bitwise the FlashInfer RMSNorm."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rmsnorm_release import (
    k31_rmsnorm_release as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "the RMSNorm requires a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(rows, cols, row_stride, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    wide = torch.randn(rows, row_stride, generator=g, device="cuda").to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(cols, generator=g, device="cuda")).to(torch.bfloat16)
    return wide[:, :cols], w


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    s = cell.spec
    x, w = _build(s["rows"], s["cols"], s["row_stride"], zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(x, w, 1e-6)
    ref = op.reference(x, w, 1e-6)
    assert out.is_contiguous() and out.shape == ref.shape
    assert torch.equal(out.view(torch.int16), ref.contiguous().view(torch.int16))


def test_under_graph_before_a_planned_gemm() -> None:
    """Released norm -> planned GEMM in one CUDA graph: the GEMM reads the norm's output, not stale memory."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
        k31_low_m_linear,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
        K31_LOW_M_PLANS,
    )

    plan = K31_LOW_M_PLANS["tep16"]
    x, w = _build(16, 1536, 3040, seed=3)
    wq = (torch.randn(1152, 1536, device="cuda") / 40).to(torch.bfloat16)
    ref = torch.nn.functional.linear(torch.ops.trtllm.flashinfer_rmsnorm(x, w, 1e-6), wq)
    out = {}

    def step():
        out["y"] = k31_low_m_linear(op(x, w, 1e-6), wq, plan, True)

    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for _ in range(3):
        x.mul_(1.5)  # new values each replay, read through the released chain
        ref = torch.nn.functional.linear(torch.ops.trtllm.flashinfer_rmsnorm(x, w, 1e-6), wq)
        g.replay()
        torch.cuda.synchronize()
        scale = ref.float().abs().amax(dim=1, keepdim=True)
        assert torch.all((out["y"].float() - ref.float()).abs() <= scale * 2.0**-7 + 1e-6)


def test_guard_refuses_a_misaligned_row_stride() -> None:
    x, w = _build(4, 1536, 1539, seed=1)
    with pytest.raises(AssertionError):
        with validating(op):
            op(x, w, 1e-6)
