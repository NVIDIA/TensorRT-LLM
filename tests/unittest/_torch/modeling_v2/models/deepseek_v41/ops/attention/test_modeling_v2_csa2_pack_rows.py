# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_pack_rows / csa2_unpack_rows entries: the three byte-row formats of the CSA2 pools.

Both ops are deterministic byte logic, so each cell gates the op against its brute-force reference
bit for bit (no band): the reference finds the nearest e2m1 code by distance to the code table and
takes the scales from float64 `ceil(log2(.))` corrected by exact comparisons (the bare `log2` of a
float64 CUDA tensor is a hair off at exact powers of two), where the op uses `frexp` and a threshold
ladder.
"""

import math

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    _E2M1,
    csa2_pack_rows,
    csa2_unpack_rows,
    row_bytes,
)

WIDTH = {"swa": 512, "main": 512, "index": 128}


def _x(spec, cell):
    gen = gen_for(cell)
    n = WIDTH[spec["kind"]]
    x = torch.randn(spec["rows"], n, generator=gen, device=DEV)
    x = x * torch.rand(spec["rows"], 1, generator=gen, device=DEV) * spec["amp"]
    return x.bfloat16()


@pytest.mark.parametrize(
    "cell", csa2_pack_rows.CELLS, ids=[c.why[:44] for c in csa2_pack_rows.CELLS]
)
def test_pack_cells(cell, certifying):  # noqa: F811
    certifying(csa2_pack_rows)
    x = _x(cell.spec, cell)
    got = csa2_pack_rows(x, cell.spec["kind"])
    assert got.dtype == torch.uint8 and got.shape == (x.shape[0], row_bytes(cell.spec["kind"]))
    assert torch.equal(got, csa2_pack_rows.reference(x, cell.spec["kind"]))


@pytest.mark.parametrize(
    "cell", csa2_unpack_rows.CELLS, ids=[c.why[:44] for c in csa2_unpack_rows.CELLS]
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_unpack_cells(cell, out_dtype, certifying):  # noqa: F811
    certifying(csa2_unpack_rows)
    kind = cell.spec["kind"]
    packed = csa2_pack_rows.reference(
        _x(cell.spec, cell), kind
    )  # bytes from the reference, not the op
    got = csa2_unpack_rows(packed, kind, out_dtype)
    assert got.dtype == out_dtype
    assert torch.equal(got, csa2_unpack_rows.reference(packed, kind, out_dtype))


def test_row_bytes():
    assert (row_bytes("swa"), row_bytes("main"), row_bytes("index")) == (528, 288, 68)


@pytest.mark.parametrize("kind", ["swa", "main", "index"])
def test_roundtrip_is_idempotent_and_close(kind):
    torch.manual_seed(0)
    x = (torch.randn(64, WIDTH[kind], device=DEV) * torch.rand(64, 1, device=DEV) * 8).bfloat16()
    with validating(csa2_pack_rows, csa2_unpack_rows):
        y = csa2_unpack_rows(csa2_pack_rows(x, kind), kind, torch.float32)
        # a second pass is a fixed point
        assert torch.equal(csa2_unpack_rows(csa2_pack_rows(y, kind), kind, torch.float32), y)
    rel = (y - x.float()).norm() / x.float().norm()
    assert rel < (0.04 if kind == "swa" else 0.2), rel


def test_swa_matches_reference_formula():
    torch.manual_seed(1)
    x = torch.randn(8, 512, device=DEV)
    xb = x.unflatten(-1, (16, 32))
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = 2.0 ** torch.ceil(torch.log2(amax / 448.0))
    ref = ((xb / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).flatten(-2)
    assert torch.equal(csa2_unpack_rows(csa2_pack_rows(x, "swa"), "swa", torch.float32), ref)


def test_exact_ties_round_to_even():
    # the block's amax is 6, so the ue8m0 scale is exactly 1 and the inputs are the e2m1 midpoints
    row = torch.zeros(1, 128, device=DEV)
    row[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    got = csa2_unpack_rows(csa2_pack_rows(row, "index"), "index", torch.float32)[0, :8]
    assert got.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    assert math.isfinite(got.sum().item())
    assert tuple(_E2M1) == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.mark.parametrize(
    "x,kind",
    [
        (torch.zeros(2, 100), "swa"),  # wrong width
        (torch.zeros(2, 512), "bogus"),  # unknown kind
        (torch.zeros(2, 512, dtype=torch.int32), "swa"),  # integers are not values
    ],
)
def test_pack_rejections(x, kind):
    with pytest.raises(AssertionError):
        csa2_pack_rows.is_valid(x, kind)


@pytest.mark.parametrize(
    "buf,kind",
    [
        (torch.zeros(2, 100, dtype=torch.uint8), "swa"),  # wrong row width
        (torch.zeros(2, 528), "swa"),  # not bytes
        (torch.zeros(2, 528, dtype=torch.uint8), "bogus"),
    ],
)
def test_unpack_rejections(buf, kind):
    with pytest.raises(AssertionError):
        csa2_unpack_rows.is_valid(buf, kind)
