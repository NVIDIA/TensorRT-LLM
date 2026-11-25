# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the clamped_swiglu entry: the shared expert's SwiGLU with the checkpoint's training clamps.

Band: the op is fp32 `silu(g) * u`; the reference is the float64 `g * sigmoid(g) * u` with explicit
min / max clamps. One fp32 rounding per elementwise op (silu, product) => rtol 1e-5, atol 1e-6.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.activation.clamped_swiglu import (  # noqa: E501
    clamped_swiglu,
)


@pytest.mark.parametrize(
    "cell", clamped_swiglu.CELLS, ids=[c.why[:44] for c in clamped_swiglu.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(clamped_swiglu)
    s = cell.spec
    gen = gen_for(cell)
    gate = (torch.randn(s["shape"], generator=gen, device=DEV) * 12).to(
        s["dtype"]
    )  # reaches the clamp
    up = (torch.randn(s["shape"], generator=gen, device=DEV) * 12).to(s["dtype"])
    got, ref = clamped_swiglu(gate, up, s["limit"]), clamped_swiglu.reference(gate, up, s["limit"])
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
    if s["limit"] > 0:
        assert (gate.float() > s["limit"]).any() and (up.float().abs() > s["limit"]).any()


def test_swiglu_clamps():
    gate = torch.tensor([100.0, -100.0, 1.0], device=DEV)
    up = torch.tensor([100.0, -100.0, 1.0], device=DEV)
    with validating(clamped_swiglu):
        out = clamped_swiglu(gate, up, 10.0)
        free = clamped_swiglu(gate, up, 0.0)
    silu = torch.nn.functional.silu
    ref = silu(torch.tensor([10.0, -100.0, 1.0], device=DEV)) * torch.tensor(
        [10.0, -10.0, 1.0], device=DEV
    )
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(free, silu(gate) * up)


def test_rejections():
    x = torch.zeros(2, 4, device=DEV)
    clamped_swiglu.is_valid(x, x, 0.0)
    with pytest.raises(AssertionError, match="negative"):
        clamped_swiglu.is_valid(x, x, -1.0)
    with pytest.raises(AssertionError):
        clamped_swiglu.is_valid(x, torch.zeros(2, 3, device=DEV), 1.0)
