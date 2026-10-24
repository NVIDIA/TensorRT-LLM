# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the beam_depth_residual entry."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.beam_depth_residual import (
    beam_depth_residual as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_depth_residual requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (spec["tokens"], spec["hidden"])
    branch = torch.randn(shape, generator=g, device="cuda").to(spec["dtype"])
    residual = (2.0 * torch.randn(shape, generator=g, device="cuda")).to(spec["dtype"])
    return branch, residual


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Bitwise: two scalings and an add, each an fp32 operation rounded once."""
    certifying(op)
    branch, residual = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    branch_scale, residual_scale = cell.spec["scales"]
    out = op(branch, residual, branch_scale, residual_scale)
    assert out.shape == branch.shape and out.dtype == branch.dtype
    ref = op.reference(branch, residual, branch_scale, residual_scale)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_scales_are_the_source_depth_coefficients() -> None:
    """The cells carry blocks 0 and 3's coefficients: the n-th residual add scales by
    (1, sqrt(n - 1)) / sqrt(n); block i's attention add is n = 2(i + 1), its feed-forward
    add n = 2(i + 1) + 1."""
    scales = {cell.why.split(",")[0]: cell.spec["scales"] for cell in op.CELLS}
    assert scales["block 0 attention add"] == (1 / math.sqrt(2), 1 / math.sqrt(2))
    assert scales["block 0 feed-forward add"] == (1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3))
    assert scales["block 3 attention add"] == (1 / math.sqrt(8), math.sqrt(7) / math.sqrt(8))
    assert scales["block 3 feed-forward add"] == (1 / 3, math.sqrt(8) / 3)
    for block, (attention, feed_forward) in ((0, (2, 3)), (3, (8, 9))):
        for kind, n in (("attention", attention), ("feed-forward", feed_forward)):
            branch, residual = scales[f"block {block} {kind} add"]
            assert (branch, residual) == (1 / math.sqrt(n), math.sqrt(n - 1) / math.sqrt(n))


def test_fused_fp32_add_is_a_different_answer() -> None:
    branch, residual = _build(dict(tokens=129, hidden=5120, dtype=torch.bfloat16), seed=5)
    a, b = 1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3)
    fused = (branch.float() * a + residual.float() * b).to(torch.bfloat16)
    assert not torch.equal(op(branch, residual, a, b), fused)


@pytest.mark.parametrize(
    "case,match",
    [
        ("broadcast_residual", "must match"),
        ("fp32_residual", "share one dtype"),
        ("tensor_scale", "Python floats"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    branch, residual = _build(dict(tokens=17, hidden=5120, dtype=torch.bfloat16), seed=3)
    branch_scale, residual_scale = 1 / math.sqrt(2), 1 / math.sqrt(2)
    if case == "broadcast_residual":
        residual = residual[:1]
    elif case == "fp32_residual":
        residual = residual.float()
    else:
        branch_scale = torch.tensor(branch_scale, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(branch, residual, branch_scale, residual_scale)
