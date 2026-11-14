# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 gemma_add_rmsnorm_1row entry.

The kernel replays the stock kernel's row sum, so it is gated bitwise against the stock
gemma add+norm on the same inputs, and against the fp32 reference in the stock norm's
band (the residual bitwise, the output in torch's default bf16 band). Then CUDA-graph
replay at the decode bucket the target captures, and the aliasing guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.flashinfer_gemma_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_gemma_fused_add_rmsnorm as stock,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.gemma_add_rmsnorm_1row import (  # noqa: E501
    gemma_add_rmsnorm_1row as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """`(x, residual, weight, eps)`: rows at magnitudes spread over e^[-6, 5], one zero and
    one -0.0 16-column block per row, a weight with entries at exactly -1 and at x20."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, hidden = spec["rows"], spec["hidden"]
    scale = torch.empty(rows, 1, device="cuda").uniform_(-6.0, 5.0, generator=g).exp()
    x = (torch.randn(rows, hidden, generator=g, device="cuda") * scale).to(spec["dtype"])
    residual = (torch.randn(rows, hidden, generator=g, device="cuda") * 4 * scale).to(spec["dtype"])
    x[:, 32:48] = 0.0
    residual[:, 32:48] = 0.0
    x[:, 80:96] = -0.0
    residual[:, 80:96] = -0.0
    weight = torch.randn(hidden, generator=g, device="cuda") * 0.3
    weight[torch.randperm(hidden, generator=g, device="cuda")[: hidden // 100]] = -1.0
    weight[torch.randperm(hidden, generator=g, device="cuda")[: hidden // 100]] *= 20.0
    return x, residual, weight.to(spec["dtype"]), spec["eps"]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x, residual, weight, eps = _build(cell.spec, seed=_seed(cell.why))
    stock_x, stock_residual = stock(x.clone(), residual.clone(), weight, eps)
    ref_x, ref_residual = op.reference(x, residual, weight, eps)

    out_x, out_residual = op(x, residual, weight, eps)

    assert out_x is x and out_residual is residual, "the op writes in place"
    assert torch.equal(_bits(x), _bits(stock_x)), "x differs from the stock kernel's"
    assert torch.equal(_bits(residual), _bits(stock_residual)), "residual differs from stock"
    assert torch.equal(_bits(residual), _bits(ref_residual))
    torch.testing.assert_close(x, ref_x)


def test_cuda_graph_replay() -> None:
    """The decode graph at 128 rows: replays with fresh inputs equal the stock kernel."""
    spec = op.CELLS[-1].spec
    x, residual, weight, eps = _build(spec, seed=_seed("graph"))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x.clone(), residual.clone(), weight, eps)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, residual, weight, eps)

    for replay in range(3):
        fresh_x, fresh_residual, _, _ = _build(spec, seed=_seed(f"replay {replay}"))
        want_x, want_residual = stock(fresh_x.clone(), fresh_residual.clone(), weight, eps)
        x.copy_(fresh_x)
        residual.copy_(fresh_residual)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(x), _bits(want_x)), f"replay {replay}: x"
        assert torch.equal(_bits(residual), _bits(want_residual)), f"replay {replay}: residual"


def test_guard_refuses_aliased_buffers() -> None:
    """One buffer passed as both: the op writes the sum, then the output, over it."""
    x, _, weight, eps = _build(op.CELLS[0].spec, seed=_seed("alias"))
    with pytest.raises(AssertionError, match="distinct buffers"):
        with validating(op):
            op(x, x, weight, eps)
