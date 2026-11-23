# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bmm_out__r16 entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.bmm_out__r16 import (  # noqa: E501
    bmm_out as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"

CELLS = op.CELLS


def _build(spec, seed):
    """`(a, b, out, parent)`; with `transposed_out`, `out` is a transposed view of `parent` as the target passes it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, m, k, n, dt = spec["batch"], spec["m"], spec["k"], spec["n"], spec["dtype"]
    a = torch.randn(b, m, k, generator=g, device="cuda").to(dt)
    w = torch.randn(b, k, n, generator=g, device="cuda").to(dt)
    if spec["transposed_out"]:
        parent = torch.zeros(m, b, n, dtype=dt, device="cuda")
        out = torch.transpose(parent, 0, 1)
    else:
        parent = torch.zeros(b, m, n, dtype=dt, device="cuda")
        out = parent
    return a, w, out, parent


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    a, w, out, parent = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    assert op(a, w, out) is out
    # The product must land in the caller's buffer, not a re-allocation.
    assert out.data_ptr() == parent.data_ptr()
    # Identical operands; only the K summation order differs.
    assert_within_ulp(out, op.reference(a, w, out), element_ulp=4.0, rms_ulp=2.0)
