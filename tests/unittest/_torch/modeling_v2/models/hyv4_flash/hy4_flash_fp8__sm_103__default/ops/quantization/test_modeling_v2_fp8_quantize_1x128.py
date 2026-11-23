# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_quantize_1x128 entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "fp8_quantize_1x128 requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="fp8_quantize_1x128 is certified on sm_103"
)


def _ordinal(codes: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude ordinal of an e4m3 byte: adjacent codes differ by 1."""
    b = codes.view(torch.uint8).int()
    return torch.where(b >= 128, -(b & 0x7F), b & 0x7F)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()))
    x = (torch.randn(spec["rows"], spec["cols"], device="cuda", generator=g) * 3.0).to(
        torch.bfloat16
    )
    data, scales = op(x, spec["use_ue8m0"])
    ref_data, ref_scales = op.reference(x, spec["use_ue8m0"])
    assert data.dtype == ref_data.dtype and data.shape == ref_data.shape and data.is_contiguous()
    assert (
        scales.dtype == ref_scales.dtype
        and scales.shape == ref_scales.shape
        and scales.is_contiguous()
    )
    # The default build divides exactly (bit-exact); the band admits a fast-math build, whose two divisions are
    # each within 2 ulp, so a code can move at most one e4m3 step and only at a rounding tie.
    scale_ulp = (
        (scales.view(torch.int32).long() - ref_scales.view(torch.int32).long()).abs().max().item()
    )
    assert scale_ulp <= 4, f"scale {scale_ulp} fp32 ulp off"
    step = (_ordinal(data) - _ordinal(ref_data)).abs()
    assert step.max().item() <= 1, f"a code is {step.max().item()} e4m3 steps off"
    assert (step == 0).float().mean().item() >= 0.995, "fewer than 99.5% of codes bit-exact"
