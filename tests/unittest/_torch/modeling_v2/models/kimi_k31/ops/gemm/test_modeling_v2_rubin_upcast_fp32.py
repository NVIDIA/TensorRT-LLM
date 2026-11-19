# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_upcast_fp32, BF16 rows widened to FP32 bit for bit.

GPU (sm_107): a sample of the cells bit for bit against the reference (``x.float()``).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_upcast_fp32 import (  # noqa: E501
    rubin_upcast_fp32 as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

on_sm107 = on_sm(107)


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@on_sm107
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(s["rows"])
    x = torch.randint(-(2**15), 2**15, (s["rows"], s["cols"]), generator=gen, device="cuda")
    x = x.to(torch.int16).view(torch.bfloat16)
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    out = op(x)
    assert out.dtype == torch.float32 and out.shape == x.shape and out.is_contiguous()
    assert torch.equal(out.view(torch.int32), op.reference(x).view(torch.int32))
