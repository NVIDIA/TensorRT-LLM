# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m31_mxfp8_gemm entry of the MiniMax-M3.1-Flash sm_107 targets (the decode-graph GEMM on a
prequantized activation)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.m31_mxfp8_gemm import (  # noqa: E501
    m3_mxfp8_gemm as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

# The targets share some cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


@pytest.mark.parametrize("cell", CELLS, ids=[f"{c.why[:40]}-k{c.spec['k']}" for c in CELLS])
def test_certified_cells(cell) -> None:
    """Same E4M3 operands and power-of-two scales on both sides; accumulation order then one bf16 rounding: at most
    2 units of ULP[bf16] of the row (the low-M GEMM's band), RMS well under one."""
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    act, act_sf = mxfp8(spec["m"], spec["k"], g)
    w, w_sf = mxfp8(spec["n"], spec["k"], g)
    args = (act, act_sf, w, w_sf, torch.bfloat16, False)
    with validating(op):
        out = op(*args)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype == torch.bfloat16
    assert_within_ulp(out, op.reference(*args), element_ulp=2.0, rms_ulp=0.25)
