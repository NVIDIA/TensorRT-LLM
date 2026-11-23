# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gated_attn_output entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.activation.gated_attn_output import (  # noqa: E501
    gated_attn_output as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "gated_attn_output requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()))
    out = torch.randn(spec["rows"], spec["width"], generator=g, device="cuda").to(spec["dtype"])
    gate = torch.randn(spec["rows"], spec["width"], generator=g, device="cuda").to(spec["dtype"])
    ref = op.reference(out, gate)
    result = op(out, gate)
    assert result.data_ptr() == out.data_ptr()
    # `tl.sigmoid`'s fast exp may flip a bf16 rounding tie against torch's fp32 sigmoid.
    assert_within_ulp(result, ref, element_ulp=2.0, rms_ulp=1.0)
