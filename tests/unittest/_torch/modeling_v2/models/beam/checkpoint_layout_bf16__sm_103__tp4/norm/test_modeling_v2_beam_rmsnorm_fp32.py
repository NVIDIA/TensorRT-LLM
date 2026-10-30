# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the beam_rmsnorm_fp32 entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.beam_rmsnorm_fp32 import (  # noqa: E501
    beam_rmsnorm_fp32 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_rmsnorm_fp32 requires a CUDA device"

SEEDS = (17, 41)


def _build(spec, seed):
    """Hidden rows with a residual stream's spread (a few large channels), a weight near 1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(spec["tokens"], spec["width"], generator=g) * 4.0
    x[:, :: spec["width"] // 8] *= 64.0
    w = 1.0 + 0.25 * torch.randn(spec["width"], generator=g)
    return x.to(spec["dtype"]).cuda(), w.to(spec["dtype"]).cuda()


def _bf16_units_apart(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance between same-signed bf16 values in units of the last place."""
    assert a.dtype == b.dtype == torch.bfloat16
    assert bool(((a.float() >= 0) == (b.float() >= 0)).all()), "signs differ"
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:26] for c in op.CELLS])
def test_certified_cells(cell, seed, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Within one bf16 unit of the fp64 reference: the fp32 steps carry a few fp32 units of
    error, so the single rounding at the end may land on either neighbour of a boundary."""
    certifying(op)
    x, w = _build(cell.spec, seed)
    before = x.clone()
    out = op(x, w, cell.spec["eps"])
    ref = op.reference(x, w, cell.spec["eps"])
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    assert out.data_ptr() != x.data_ptr() and torch.equal(x, before)
    assert int(_bf16_units_apart(out, ref).max()) <= 1


def test_rounds_once_at_the_end() -> None:
    """Rounding the normalized row to bf16 before the weight is a different answer: it
    rounds twice, and on these rows some outputs move by a bf16 unit."""
    spec = op.CELLS[2].spec
    x, w = _build(spec, seed=17)
    out = op(x, w, spec["eps"])
    x32 = x.float()
    rms = x32.pow(2).mean(dim=-1, keepdim=True).add(spec["eps"]).rsqrt()
    rounded_twice = (x32 * rms).to(torch.bfloat16) * w
    assert torch.equal(out, (x32 * rms * w).to(torch.bfloat16))
    assert bool((out != rounded_twice).any())


@pytest.mark.parametrize(
    "case,match",
    [
        ("fp32_weight", "must be torch.bfloat16"),
        ("narrow_weight", "weight must be"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    x, w = _build(op.CELLS[1].spec, seed=17)
    w = w.float() if case == "fp32_weight" else w[:1]
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(x, w, 1e-6)
