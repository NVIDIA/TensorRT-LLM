# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the layer_norm entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.norm.layer_norm import (  # noqa: E501
    layer_norm as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import ULP, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "layer_norm requires a CUDA device"

# The range of the checkpoint's vision LayerNorm weights and biases: wide enough that a wrong
# normalization is not masked by a near-unit affine.
WEIGHT_LO, WEIGHT_HI = -0.78125, 14.3125
BIAS_LO, BIAS_HI = -19.5, 10.9375


def _build(spec, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=gen, device="cuda").to(dtype)
    weight = (
        torch.rand(shape[-1], generator=gen, device="cuda") * (WEIGHT_HI - WEIGHT_LO) + WEIGHT_LO
    )
    bias = torch.rand(shape[-1], generator=gen, device="cuda") * (BIAS_HI - BIAS_LO) + BIAS_LO
    return x, weight.to(dtype), bias.to(dtype), spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, weight, bias, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, weight, bias, eps)
    ref = op.reference(x, weight, bias, eps)
    torch.testing.assert_close(out, ref)
    # Fp32 kernel vs fp64 reference: also gate the aggregate error, not just each element.
    of, rf = out.float(), ref.float()
    rel_rms = (of - rf).pow(2).mean().sqrt() / rf.pow(2).mean().sqrt().clamp_min(1e-9)
    assert rel_rms <= 0.05 * ULP[torch.bfloat16], (
        f"relative RMS {rel_rms / ULP[torch.bfloat16]:.4f} ulp"
    )
