# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gelu_erf catalog entry.

The entry's identity is *which* GELU it is, and the wrong answer -- the tanh
GELU this same tower's block MLPs use -- rounds to within about one bf16 ulp of
it; only fp32 tells the two apart.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.activation.gelu_erf import (  # noqa: E501
    gelu_erf as op,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "gelu_erf requires a CUDA device"

MERGER_WIDTH = 4608
TOWER_HIDDEN = 1152
#: model.visual.merger, measured from the checkpoint: linear_fc1.weight std
#: 0.010927, linear_fc1.bias std 0.002407, norm.weight mean 1.174581 / std
#: 0.140536. The real tensors give an activand at std 0.875 / absmax 4.23.
FC1_WEIGHT_STD = 0.010927
FC1_BIAS_STD = 0.002407
NORM_GAMMA_MEAN = 1.174581
NORM_GAMMA_STD = 0.140536

SQRT_2 = math.sqrt(2.0)
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
    if spec.get("dist", "uniform") == "merger":
        g = torch.Generator(device="cuda").manual_seed(seed)
        rows = shape[0]
        gamma = NORM_GAMMA_MEAN + NORM_GAMMA_STD * torch.randn(
            TOWER_HIDDEN, generator=g, device="cuda"
        )
        hidden = torch.randn(4 * rows, TOWER_HIDDEN, generator=g, device="cuda")
        grouped = (hidden * gamma.to(dtype).float()).reshape(rows, MERGER_WIDTH)
        weight = FC1_WEIGHT_STD * torch.randn(
            MERGER_WIDTH, MERGER_WIDTH, generator=g, device="cuda"
        )
        bias = FC1_BIAS_STD * torch.randn(MERGER_WIDTH, generator=g, device="cuda")
        return (grouped @ weight.to(dtype).float().t() + bias.to(dtype).float()).to(dtype)
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


def _erf64(x):
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.erf(x64 / SQRT_2))


def _tanh64(x):
    """The *other* GELU -- `gelu_pytorch_tanh`, this tower's block-MLP activation."""
    x64 = x.double()
    return 0.5 * x64 * (1.0 + torch.tanh(SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)))
