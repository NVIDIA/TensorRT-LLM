# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp gelu_tanh entry.

The entry's identity is *which* GELU it is: in bf16 the tanh and erf GELUs
round to within about one ulp of each other, and only fp32 tells them apart.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.activation.gelu_tanh import (  # noqa: E501
    gelu_tanh as op,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gelu_tanh requires a CUDA device"

INTERMEDIATE_SIZE = 4304
HIDDEN_SIZE = 1152
#: model.visual.blocks.*.mlp.linear_fc1, measured from the checkpoint: weight
#: std 0.0197 (block 0), bias std 1.676 / absmax 9.44 -- the bias dominates.
FC1_WEIGHT_STD = 0.0197
FC1_BIAS_STD = 1.676

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

#: torch.testing's untouched per-dtype defaults, restated only so the
#: discrimination test can say how far outside that gate each variant lands.
_RTOL = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
_ATOL = 1e-5


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _uniform(shape, lo, hi, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(lo, hi, generator=g)
    return x.to(dtype)


def _build(spec, seed):
    shape, dtype = spec["shape"], spec["dtype"]
    if spec.get("dist", "uniform") == "fc1":
        g = torch.Generator(device="cuda").manual_seed(seed)
        rows = shape[0]
        hidden = torch.randn(rows, HIDDEN_SIZE, generator=g, device="cuda").to(dtype)
        weight = (
            FC1_WEIGHT_STD * torch.randn(shape[-1], HIDDEN_SIZE, generator=g, device="cuda")
        ).to(dtype)
        bias = (FC1_BIAS_STD * torch.randn(shape[-1], generator=g, device="cuda")).to(dtype)
        return (hidden.float() @ weight.float().t() + bias.float()).to(dtype)
    lo, hi = spec["lo"], spec["hi"]
    return _uniform(shape, lo, hi, dtype, seed)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x = _build(cell.spec, seed=_seed(cell.why))
    before = x.clone()
    out = op(x)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert out.data_ptr() != x.data_ptr(), "the entry is out-of-place"
    assert torch.equal(before, x), "the entry must not mutate its input"
    torch.testing.assert_close(out, op.reference(x))


def _tanh64(x):
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)))


def _erf64(x):
    """The *other* GELU -- exact, erf-based; `torch.nn.GELU()` and `moe_gelu`."""
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.erf(x64 / math.sqrt(2.0)))
