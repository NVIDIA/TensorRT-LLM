# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 grouped-convolution entry (CuTe DSL).

A sample of the certified cells is gated bit for bit against the entry's reference (the target's torch
convolution): same bytes, signed zeros and NaN payloads included. The inputs are laid out as the
drafter produces them: the delta is one side's strided view of the kernel projection's
``[rows, 2 * taps * groups]`` output, the base kernel one side of ``[2, taps, hidden]``.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_grouped_conv import (  # noqa: E501
    dflash2_grouped_conv as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

pytestmark = on_sm(100, 103, 107, reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x")


def _build_raw(spec, seed):
    """``(x, projection, base)``: activations, kernel projection output, both sides' base."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch, width = spec["batch"], spec["width"]
    hidden, taps, group = spec["hidden"], spec["taps"], spec["group"]
    rows, groups = batch * width, hidden // group
    x = torch.randn(rows, hidden, generator=g, device="cuda")
    projection = 0.25 * torch.randn(rows, 2 * taps * groups, generator=g, device="cuda")
    base = 0.5 * torch.randn(2, taps, hidden, generator=g, device="cuda")
    if spec["fill"] == "nan_prev":
        # Request 2 is a padding request full of NaN; request 3's block starts right after it.
        x[2 * width : 3 * width] = float("nan")
        projection[2 * width : 3 * width] = float("nan")
    elif spec["fill"] == "zeros":
        keep = torch.rand(rows, hidden, generator=g, device="cuda") < 0.5
        sign = torch.where(torch.rand(rows, hidden, generator=g, device="cuda") < 0.5, 1.0, -1.0)
        x = torch.where(keep, x, 0.0 * sign)
        projection = -projection.abs()
        base = -base.abs()
    return x.to(torch.bfloat16), projection.to(torch.bfloat16), base.to(torch.bfloat16)


def _args(spec, x, projection, base):
    """The op's arguments: one side's delta view and base kernel, as the drafter passes them."""
    taps, groups = spec["taps"], spec["hidden"] // spec["group"]
    delta = projection.view(x.shape[0], 2, taps, groups)[:, spec["side"]]
    return x, delta, base[spec["side"]], spec["width"], spec["group"]


def _build(spec, seed):
    return _args(spec, *_build_raw(spec, seed))


CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    assert torch.equal(op(*args).view(torch.int16), op.reference(*args).view(torch.int16))
