# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gelu entry."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.activation.gelu import (  # noqa: E501
    gelu as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "gelu requires a CUDA device"

MLP_WIDTH = 4304
MERGER_WIDTH = 4608
TOWER_HIDDEN = 1152
SPATIAL_MERGE = 4  # 2 x 2

# Layer-0 weight statistics of the checkpoint's vision tower.
FC1_WEIGHT_STD = 0.018955
FC1_BIAS_MEAN = -2.416808
FC1_BIAS_STD = 1.668815
LN_GAMMA_MEAN = 1.511802
LN_GAMMA_STD = 2.353820
PROJ_WEIGHT_STD = 0.008602
PROJ_BIAS_MEAN = -0.000401
PROJ_BIAS_STD = 0.003034
PERCEIVE_GAMMA_MEAN = 1.084490
PERCEIVE_GAMMA_STD = 0.029593


def _uniform(shape, lo, hi, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(lo, hi, generator=g)
    return x.to(dtype)


def _tower(rows, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gamma = LN_GAMMA_MEAN + LN_GAMMA_STD * torch.randn(TOWER_HIDDEN, generator=g, device="cuda")
    hidden = torch.randn(rows, TOWER_HIDDEN, generator=g, device="cuda")
    weight = FC1_WEIGHT_STD * torch.randn(MLP_WIDTH, TOWER_HIDDEN, generator=g, device="cuda")
    bias = FC1_BIAS_MEAN + FC1_BIAS_STD * torch.randn(MLP_WIDTH, generator=g, device="cuda")
    return ((hidden * gamma) @ weight.t() + bias).to(dtype)


def _merger(rows, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gamma = PERCEIVE_GAMMA_MEAN + PERCEIVE_GAMMA_STD * torch.randn(
        TOWER_HIDDEN, generator=g, device="cuda"
    )
    hidden = torch.randn(rows, SPATIAL_MERGE, TOWER_HIDDEN, generator=g, device="cuda")
    hidden = (hidden * gamma).reshape(rows, MERGER_WIDTH)
    weight = PROJ_WEIGHT_STD * torch.randn(MERGER_WIDTH, MERGER_WIDTH, generator=g, device="cuda")
    bias = PROJ_BIAS_MEAN + PROJ_BIAS_STD * torch.randn(MERGER_WIDTH, generator=g, device="cuda")
    return (hidden @ weight.t() + bias).to(dtype)


def _build(spec, seed):
    shape, dtype = spec["shape"], spec["dtype"]
    dist = spec.get("dist", "uniform")
    if dist == "tower":
        return _tower(shape[0], dtype, seed)
    if dist == "merger":
        return _merger(shape[0], dtype, seed)
    return _uniform(shape, spec["lo"], spec["hi"], dtype, seed)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x)
    assert out.data_ptr() != x.data_ptr(), "the entry is out-of-place"
    # fp32 kernel vs fp64 reference: only flipped bf16 rounding ties differ.
    assert_within_ulp(out, op.reference(x), element_ulp=2.0, rms_ulp=0.1)


def test_identity_is_the_erf_form_and_not_tanh() -> None:
    """In bf16 the tanh GELU is within about one ulp of the erf GELU, so only fp32 tells them apart."""
    x = _uniform((512, MLP_WIDTH), -4.0, 4.0, torch.float32, 24)
    out = op(x)
    torch.testing.assert_close(out, op.reference(x))
    x64 = x.double()
    tanh = 0.5 * x64 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x64 + 0.044715 * x64**3)))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out, tanh.float())
