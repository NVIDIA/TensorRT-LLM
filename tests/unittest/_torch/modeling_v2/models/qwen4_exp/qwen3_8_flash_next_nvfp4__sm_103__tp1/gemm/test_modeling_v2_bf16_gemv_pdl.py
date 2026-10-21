# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bf16_gemv_pdl entry.

Cells against the fp32 reference within the band `low_m_bf16_mm` uses (the
same products, summed in another order). Beyond the cells: a PDL chain under
CUDA-graph capture and replay, where each GEMV consumes the previous one's
output and so would read a stale activation if it read before its wait; and
the layout guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.bf16_gemv_pdl import (  # noqa: E501
    bf16_gemv_pdl as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "bf16_gemv_pdl requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Accumulation order only: 4 bf16 ulp element-wise against the row scale,
    2 ulp of relative RMS, as for `low_m_bf16_mm` and `cublas_mm`."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _build(m, n, k, seed):
    """`mat_b` is a `[N, K]` weight's `.t()`, as the target stores its projections."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    return mat_a, weight.t()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    mat_a, mat_b = _build(s["m"], s["n"], s["k"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, s["pdl"])
    assert out.shape == (s["m"], s["n"]) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _compare(out, op.reference(mat_a, mat_b, s["pdl"]))


def test_dependent_chain_under_graph_replay() -> None:
    """Decode runs this under CUDA graphs, back to back with other kernels. Each
    link of the chain reads the output of the one before it: a GEMV that read
    its activation before `griddepcontrol.wait` would see the previous replay's
    values, and the comparison below would fail on the second replay."""
    pdl = True
    dims = [2560, 6144, 2560, 640, 2560]  # K of each link; N is the next link's K
    weights = [_build(1, n, k, seed=i)[1] for i, (k, n) in enumerate(zip(dims, dims[1:] + [16480]))]
    x = torch.randn(1, dims[0], device="cuda").to(torch.bfloat16)

    def chain():
        h = x
        outs = []
        for w in weights:
            h = op(h, w, pdl)
            outs.append(h)
        return outs

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            chain()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain()
    for trial in range(3):
        g = torch.Generator(device="cuda").manual_seed(100 + trial)
        x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(x.dtype))
        graph.replay()
        torch.cuda.synchronize()
        h = x
        for w, out in zip(weights, outs):
            # Each link against the reference of what the previous link actually stored.
            _compare(out, op.reference(h, w))
            h = out


def test_guard_refuses_a_row_major_mat_b() -> None:
    mat_a, mat_b = _build(1, 512, 2560, seed=3)
    with validating(op), pytest.raises(AssertionError):
        op(mat_a, mat_b.contiguous())
