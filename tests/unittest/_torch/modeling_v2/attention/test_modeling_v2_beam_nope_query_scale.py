# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the beam_nope_query_scale entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_nope_query_scale import (
    beam_nope_query_scale as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_nope_query_scale requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, h, d, dtype = spec["tokens"], spec["num_heads"], spec["head_dim"], spec["dtype"]
    q = torch.randn(t, h, d, generator=g, device="cuda").to(dtype)
    positions = torch.arange(t, device="cuda")
    return q, positions


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Bitwise: the fp32 scale and the single rounding of the product are what the
    reference spells out step by step."""
    certifying(op)
    q, positions = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(q, positions)
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out, op.reference(q, positions), rtol=0.0, atol=0.0)


def test_scale_against_float64() -> None:
    """The fp32 scale is within four fp32 units of the float64 expression at every position, so
    the bf16 result differs from a float64-scaled one only where the product sits within that
    distance of a bf16 rounding boundary."""
    positions = torch.arange(129, device="cuda")
    q = torch.ones(129, 1, 1, dtype=torch.float32, device="cuda")
    scale32 = op(q, positions).view(-1)
    scale64 = torch.exp(0.1 * torch.log1p((positions.double() + 1) / 2048))
    assert float(((scale32.double() - scale64).abs() / scale64).max()) <= 4 * 2.0**-23


def test_scale_is_visible_at_nonzero_positions() -> None:
    """Values one bf16 unit below a power of two round up under the scale once it exceeds half
    a unit -- at the later positions of a 129-token context -- while position 0's scale
    leaves them unchanged: an omitted scale is detectable only at nonzero positions."""
    positions = torch.arange(129, device="cuda")
    q = torch.full((129, 1, 1), 1.9921875, dtype=torch.bfloat16, device="cuda")
    out = op(q, positions).view(-1)
    assert float(out[0]) == 1.9921875
    assert float(out[-1]) == 2.0


def test_omitting_the_scale_or_rotating_differs() -> None:
    """A full-attention q that is left unscaled is a different tensor on ordinary inputs."""
    q, positions = _build(dict(tokens=129, num_heads=48, head_dim=128, dtype=torch.bfloat16), 3)
    assert not torch.equal(op(q, positions), q)


@pytest.mark.parametrize(
    "case,match",
    [
        ("positions_length_one", "positions must be"),
        ("float_positions", "integer token positions"),
        ("q_rank", "q must be"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    q, positions = _build(dict(tokens=17, num_heads=48, head_dim=128, dtype=torch.bfloat16), 11)
    if case == "positions_length_one":
        positions = positions[:1]
    elif case == "float_positions":
        positions = positions.float()
    else:
        q = q.view(17, 48 * 128)
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(q, positions)
