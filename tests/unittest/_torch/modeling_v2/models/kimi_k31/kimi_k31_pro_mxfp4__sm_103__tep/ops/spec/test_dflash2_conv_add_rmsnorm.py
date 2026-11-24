# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 convolution + fused add + RMSNorm entry (CuTe DSL).

Every certified cell is gated bit for bit against the entry's reference (the family convolution,
then the catalog FlashInfer fused add + RMSNorm): the normalized output and the updated residual,
eagerly and under CUDA-graph replay. The delta is the finish side's strided view of the kernel
projection's output, as the drafter passes it.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.dflash2_conv_add_rmsnorm import (  # noqa: E501
    dflash2_conv_add_rmsnorm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../../.."]
from _validating import validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM 10.0 / 10.3",
)


def _inputs(spec: dict, seed: int, scale: float):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["gens"] * spec["width"]
    hidden, taps, group = spec["hidden"], spec["taps"], spec["group"]
    x = (scale * torch.randn(rows, hidden, generator=g, device="cuda")).to(torch.bfloat16)
    x[0, :32] = 0.0
    x[-1, 32:64] = -0.0
    projection = (0.3 * torch.randn(rows, 2, taps, hidden // group, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    base = (0.5 * torch.randn(taps, hidden, generator=g, device="cuda")).to(torch.bfloat16)
    residual = torch.randn(rows, hidden, generator=g, device="cuda").to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(hidden, generator=g, device="cuda")).to(torch.bfloat16)
    return x, projection[:, 1], base, residual, weight


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
@pytest.mark.parametrize("scale", [0.05, 1.0, 30.0])
def test_certified_cells(cell, scale) -> None:
    spec = cell.spec
    x, delta, base, residual, weight = _inputs(spec, zlib.crc32(cell.why.encode()) % 2**31, scale)
    args = (x, delta, base, spec["width"], spec["group"])
    ref_out, ref_res = op.reference(*args, residual, weight, spec["eps"])
    res = residual.clone()
    with validating(op):
        out, res_out = op(*args, res, weight, spec["eps"])
    assert res_out is res
    assert _bits_equal(out, ref_out), "normalized output differs from conv + fused add-norm"
    assert _bits_equal(res, ref_res), "updated residual differs from conv + fused add-norm"

    # Graph replay (after the eager compile above) gives the same bits.
    res_g = residual.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g, _ = op(*args, res_g, weight, spec["eps"])
    res_g.copy_(residual)
    graph.replay()
    torch.cuda.synchronize()
    assert _bits_equal(out_g, ref_out) and _bits_equal(res_g, ref_res)


def test_guard_refuses_an_aliased_residual() -> None:
    spec = op.CELLS[0].spec
    x, delta, base, _, weight = _inputs(spec, 1, 1.0)
    with pytest.raises(AssertionError, match="alias"):
        with validating(op):
            op(x, delta, base, spec["width"], spec["group"], x, weight, spec["eps"])
