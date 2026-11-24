# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul__r18 entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.activation.flashinfer_silu_and_mul__r18 import (  # noqa: E501
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"

CELLS = op.CELLS


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(cell.spec["shape"], generator=g, device="cuda").to(cell.spec["dtype"])
    out = op(x)
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    # Both sides compute in fp32 and round once to bf16.
    assert_within_ulp(out, op.reference(x), element_ulp=1.0, rms_ulp=0.1)
