# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_moe_residual_add entry: bitwise against its reference on a sample of its cells on
SM107."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_residual_add import (  # noqa: E501
    HIDDEN,
    rubin_moe_residual_add,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

pytestmark = cpu_or_sm(107)
gpu = on_sm(107)


def _operands(rows, seed, device):
    """Routed and shared outputs of similar size and a larger residual, so both roundings bite."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    routed = torch.randn(rows, HIDDEN, generator=g) * 0.3
    shared = torch.randn(rows, HIDDEN, generator=g) * 0.3
    residual = torch.randn(rows, HIDDEN, generator=g) * 4.0
    return tuple(t.to(torch.bfloat16).to(device) for t in (routed, shared, residual))


CELLS = sample_cells(rubin_moe_residual_add.CELLS, size_key="rows")[:2]


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_moe_residual_add)
    any_rows = cell.spec.get("any_rows", False)
    args = _operands(cell.spec["rows"], zlib.crc32(cell.why.encode()), "cuda")
    out = rubin_moe_residual_add(*args, any_rows=any_rows)
    ref = rubin_moe_residual_add.reference(*args, any_rows=any_rows)
    # Exact: both add in FP32 and round to BF16 at the same two points.
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if cell.spec["rows"]:
        one_rounding = (args[2].float() + args[0].float() + args[1].float()).to(torch.bfloat16)
        assert not torch.equal(out, one_rounding), "the inputs do not exercise the first rounding"
