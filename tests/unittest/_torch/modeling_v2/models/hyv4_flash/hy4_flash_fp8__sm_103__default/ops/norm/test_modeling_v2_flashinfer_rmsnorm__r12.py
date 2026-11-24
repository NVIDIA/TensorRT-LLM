# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm__r12 entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.norm.flashinfer_rmsnorm__r12 import (  # noqa: E501
    flashinfer_rmsnorm as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"

CELLS = op.CELLS


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])
    weight = torch.randn(spec["shape"][-1], generator=g, device="cuda").to(spec["dtype"])
    out = op(x, weight, spec["eps"])
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    # Same fp32 math on both sides, rounded once to bf16; reduction order differs.
    assert_within_ulp(out, op.reference(x, weight, spec["eps"]), element_ulp=3.0, rms_ulp=0.5)
