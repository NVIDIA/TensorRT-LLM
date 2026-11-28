# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 convolution + fused add + RMSNorm entry (CuTe DSL).

A sample of the entry's cells (``sample_cells``) is gated bit for bit against the entry's reference
(the family convolution, then the catalog FlashInfer fused add + RMSNorm): the normalized output and
the updated residual. The delta is the finish side's strided view of the kernel projection's output,
as the drafter passes it.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.dflash2_conv_add_rmsnorm import (  # noqa: E501
    dflash2_conv_add_rmsnorm as op,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import validating  # noqa: E402

pytestmark = on_sm(100, 103, reason="the Kimi K3.1 DFlash2 kernels are certified on SM 10.0 / 10.3")


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


CELLS = sample_cells(op.CELLS, size_key="gens")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    x, delta, base, residual, weight = _inputs(spec, zlib.crc32(cell.why.encode()) % 2**31, 1.0)
    args = (x, delta, base, spec["width"], spec["group"])
    ref_out, ref_res = op.reference(*args, residual, weight, spec["eps"])
    res = residual.clone()
    with validating(op):
        out, res_out = op(*args, res, weight, spec["eps"])
    assert res_out is res
    assert _bits_equal(out, ref_out), "normalized output differs from conv + fused add-norm"
    assert _bits_equal(res, ref_res), "updated residual differs from conv + fused add-norm"
