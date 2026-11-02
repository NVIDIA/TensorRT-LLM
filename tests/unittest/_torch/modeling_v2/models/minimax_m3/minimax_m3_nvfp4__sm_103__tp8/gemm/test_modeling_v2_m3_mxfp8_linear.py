# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_mxfp8_linear entry of the MiniMax-M3 tp8 target: MXFP8 quantize + GEMM on each backend the
engine selects (native CUTLASS, FlashInfer, the per-bucket tuned quantize + GEMM)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_mxfp8_linear import (  # noqa: E501
    m3_mxfp8_linear as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mxfp8_linear requires a CUDA device"

_FLAGS = {
    "native": (False, False, False, False),
    "flashinfer": (False, False, True, False),
    "tuned": (True, False, False, False),
}


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """The activation's MXFP8 round trip is exact on both sides (same block rule); then the accumulation order and
    one bf16 rounding: at most 2 units of ULP[bf16] of the row, RMS well under one."""
    spec = cell.spec
    if (
        spec["path"] == "flashinfer"
        and getattr(torch.ops.trtllm, "flashinfer_mm_mxfp8", None) is None
    ):
        pytest.skip("this build has no trtllm::flashinfer_mm_mxfp8")
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(spec["m"], spec["k"], generator=g, device="cuda").to(torch.bfloat16)
    w, w_sf = mxfp8(spec["n"], spec["k"], g)
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    args = (x, w, w_sf, alpha, torch.bfloat16, *_FLAGS[spec["path"]])
    with validating(op):
        out = op(*args)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype == torch.bfloat16
    assert_within_ulp(out, op.reference(*args), element_ulp=2.0, rms_ulp=0.25)
