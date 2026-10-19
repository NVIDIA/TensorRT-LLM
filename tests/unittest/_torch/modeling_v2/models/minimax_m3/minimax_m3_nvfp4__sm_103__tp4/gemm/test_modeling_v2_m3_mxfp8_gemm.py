# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_mxfp8_gemm entry of the MiniMax-M3 tp4 target (the decode-graph GEMM on a prequantized
activation)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.m3_mxfp8_gemm import (  # noqa: E501
    m3_mxfp8_gemm as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mxfp8_gemm requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Same E4M3 operands and power-of-two scales on both sides; accumulation order then one bf16 rounding: at most
    2 units of ULP[bf16] of the row (the low-M GEMM's band), RMS well under one."""
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    act, act_sf = mxfp8(spec["m"], spec["k"], g)
    w, w_sf = mxfp8(spec["n"], spec["k"], g)
    with validating(op):
        out = op(act, act_sf, w, w_sf, torch.bfloat16, False)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype == torch.bfloat16
    assert_within_ulp(
        out,
        op.reference(act, act_sf, w, w_sf, torch.bfloat16, False),
        element_ulp=2.0,
        rms_ulp=0.25,
    )
