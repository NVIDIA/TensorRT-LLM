# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moe_residual_add entry: a sample of its cells bitwise against the two PyTorch
BF16 adds."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_residual_add import (
    k31_moe_residual_add as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "moe_residual_add requires a CUDA device"

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, cols = spec["rows"], spec["cols"]
    if spec["data"] == "edge":
        # Magnitudes from denormal to near the BF16 maximum, both signs: inner sums that overflow, cancel to zero,
        # and land in the denormal range, so both roundings are exercised.
        exps = torch.randint(-133, 127, (3, rows, cols), generator=g, device="cuda").float()
        mant = torch.rand(3, rows, cols, generator=g, device="cuda") + 1.0
        sign = torch.randint(0, 2, (3, rows, cols), generator=g, device="cuda").float() * 2 - 1
        vals = (sign * mant * torch.exp2(exps)).to(torch.bfloat16)
        prefix, routed, shared = vals[0].contiguous(), vals[1].contiguous(), vals[2].contiguous()
        shared[:, ::7] = -routed[:, ::7]  # exact cancellation of the inner sum
        return prefix, routed, shared
    prefix = (torch.randn(rows, cols, generator=g, device="cuda") * 8).to(torch.bfloat16)
    routed = torch.randn(rows, cols, generator=g, device="cuda").to(torch.bfloat16)
    shared = (torch.randn(rows, cols, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    return prefix, routed, shared


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    ref = op.reference(*args)
    # Bitwise: the kernel's claim is the two PyTorch adds' exact bits (inf compares equal through the int view).
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
