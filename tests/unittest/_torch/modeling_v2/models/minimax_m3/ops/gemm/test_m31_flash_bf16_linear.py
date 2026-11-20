# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m31_bf16_linear entry of the MiniMax-M3.1-Flash sm_107 targets (the drafter's
projections)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.m31_bf16_linear import (  # noqa: E501
    m3_bf16_linear as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

# The targets share some cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


@pytest.mark.parametrize("cell", CELLS, ids=[f"{c.why[:40]}-n{c.spec['n']}" for c in CELLS])
def test_certified_cells(cell) -> None:
    """Accumulation order (cuBLAS split-K) then one bf16 rounding on both sides: at most 2 units of ULP[bf16] of the
    row, RMS well under one."""
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(spec["m"], spec["k"], generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(spec["n"], spec["k"], generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    with validating(op):
        out = op(x, w)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype == torch.bfloat16
    assert_within_ulp(out, op.reference(x, w), element_ulp=2.0, rms_ulp=0.25)
