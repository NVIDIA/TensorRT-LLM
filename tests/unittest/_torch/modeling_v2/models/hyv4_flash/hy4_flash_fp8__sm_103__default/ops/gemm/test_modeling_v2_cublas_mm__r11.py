# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm__r11 entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.cublas_mm__r11 import (  # noqa: E501
    cublas_mm as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"

CELLS = op.CELLS


def _build(spec, seed):
    """`mat_b` is the `.t()` of an `[N, K]` weight: column-major, as the target passes it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, dt = spec["m"], spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(dt).t()
    out_dtype = spec["out_dtype"]
    bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt) if spec["bias"] else None
    return mat_a, mat_b, bias, out_dtype


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    # Identical operands; only the K summation order differs.
    assert_within_ulp(
        out, op.reference(mat_a, mat_b, bias, out_dtype), element_ulp=4.0, rms_ulp=2.0
    )
