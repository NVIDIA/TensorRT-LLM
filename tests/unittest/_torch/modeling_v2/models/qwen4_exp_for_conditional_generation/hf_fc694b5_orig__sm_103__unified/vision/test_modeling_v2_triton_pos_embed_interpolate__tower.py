# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp unified target's triton_pos_embed_interpolate entry
(local variant `tower`).

`op.reference` is the explicit four-corner sum in fp64, returned unrounded as
fp32, so the band measures the op's own bf16 arithmetic. `_resize64` runs
torch's `F.interpolate` in fp64 and shares no arithmetic with either: each cell
holds the reference to it (the structure: which axis is which, corner
alignment, merge row order), and its wrong variants (`align_corners=False`,
raster rows, swapped table axes) must fail the band. Beyond the cells: each
guard against a measured silent answer.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.vision.triton_pos_embed_interpolate__tower import (  # noqa: E501
    triton_pos_embed_interpolate as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "triton_pos_embed_interpolate requires a CUDA device"

NUM_GRID = 48
HIDDEN = 1152
MERGE = 2


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 bf16 ulp per element against the row scale, 1 ulp relative RMS.

    The op rounds its four corner weights to bf16 and multiply-adds in bf16 (the
    HF reference's arithmetic); `ref` is the exact value. Measured on GB300 over
    50 seeds of `_table` at the four cells: worst element 3.522 and RMS 0.752
    (128x256), p99.9 at most 1.333 ulp; the checkpoint's own table measures
    3.349 / 0.745 at most. The nearest wrong variant, `align_corners=False`,
    lands at 41.3 / 15.2 or more, raster order and swapped axes at 428 / 311
    or more.
    """
    assert out.dtype == torch.bfloat16, out.dtype
    assert_within_ulp(out.float(), ref, element_ulp=4.0, rms_ulp=1.0, ulp=ULP[torch.bfloat16])


def _table(seed: int, num_grid: int = NUM_GRID, hidden: int = HIDDEN) -> torch.Tensor:
    """A smooth `[num_grid**2, hidden]` bf16 table, the way a learned 2-D
    position table is: a 12x12 normal field upsampled bicubically, plus 0.1
    noise. Neighbouring rows have cosine 0.911; the checkpoint table's have 0.907.

    Contiguous on purpose: the upsampled field's permuted view reshapes to a
    strided table, which the kernel reads wrongly (see
    `test_guard_refuses_a_strided_table`).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    coarse = torch.randn(1, hidden, 12, 12, generator=g, device="cuda")
    up = F.interpolate(coarse, size=(num_grid, num_grid), mode="bicubic", align_corners=True)
    up = up[0].permute(1, 2, 0) + 0.1 * torch.randn(
        num_grid, num_grid, hidden, generator=g, device="cuda"
    )
    return up.reshape(num_grid * num_grid, hidden).to(torch.bfloat16).contiguous()


def _resize64(table, t, h, w, m_size, *, align=True, raster=False, swap=False):
    """torch's bilinear resize in fp64, in merge order unless `raster`."""
    num_grid = int(round(table.shape[0] ** 0.5))
    grid = table.double().view(num_grid, num_grid, -1)
    if swap:
        grid = grid.transpose(0, 1)
    src = grid.permute(2, 0, 1).unsqueeze(0).contiguous()
    out = F.interpolate(src, size=(h, w), mode="bilinear", align_corners=align)[0].permute(1, 2, 0)
    if raster:
        rows = out.reshape(h * w, -1)
    else:
        rows = (
            out.reshape(h // m_size, m_size, w // m_size, m_size, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(h * w, -1)
        )
    return rows.repeat(t, 1) if t > 1 else rows


def _args(spec, table):
    return (
        table,
        spec["t"],
        spec["h"],
        spec["w"],
        spec["num_grid_per_side"],
        spec["m_size"],
        spec["dtype"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    t, h, w, m_size = spec["t"], spec["h"], spec["w"], spec["m_size"]
    table = _table(zlib.crc32(cell.why.encode()) % 2**31, spec["num_grid_per_side"], spec["hidden"])
    before = table.clone()

    out = op(*_args(spec, table))
    assert out.shape == (t * h * w, spec["hidden"]) and out.dtype == spec["dtype"]
    assert torch.equal(table, before) and out.data_ptr() != table.data_ptr()

    ref = op.reference(*_args(spec, table))
    _compare(out, ref)

    # The reference's structure, from an implementation that shares nothing with it.
    torch.testing.assert_close(ref, _resize64(table, t, h, w, m_size).float())

    # The band refuses each wrong variant of the function.
    for wrong in (dict(align=False), dict(raster=True), dict(swap=True)):
        with pytest.raises(AssertionError):
            _compare(out, _resize64(table, t, h, w, m_size, **wrong).float())


def test_guard_refuses_a_strided_table() -> None:
    """The kernel indexes the table as dense `hidden`-wide rows: a column slice
    of a wider buffer returns, reading the wrong elements."""
    g = torch.Generator(device="cuda").manual_seed(31)
    wide = torch.randn(NUM_GRID**2, 2 * HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    strided = wide[:, :HIDDEN]
    assert not strided.is_contiguous()
    args = (1, 8, 12, NUM_GRID, MERGE, torch.bfloat16)
    unguarded = op(strided, *args)
    assert not torch.equal(unguarded, op(strided.contiguous(), *args)), "the hazard is gone"

    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(strided, *args)


def test_guard_refuses_a_grid_side_the_table_does_not_have() -> None:
    """The grid side is an argument, not read off the table: 47 on the 2304-row
    table resamples a 47x47 grid, correctly shaped and far off."""
    table = _table(32)
    args = (1, 8, 12, 47, MERGE, torch.bfloat16)
    unguarded = op(table, *args)
    assert unguarded.shape == (96, HIDDEN)
    with pytest.raises(AssertionError):
        _compare(unguarded, op.reference(table, 1, 8, 12, NUM_GRID, MERGE, torch.bfloat16))

    with pytest.raises(AssertionError, match="does not describe a 2304-row table"):
        with validating(op):
            op(table, *args)


@pytest.mark.parametrize("thw", [(1, 0, 8), (1, 8, 0), (0, 8, 8)], ids=["h0", "w0", "t0"])
def test_guard_refuses_an_empty_grid_side(thw) -> None:
    """An empty side returns no rows rather than raising."""
    table = _table(33)
    t, h, w = thw
    unguarded = op(table, t, h, w, NUM_GRID, MERGE, torch.bfloat16)
    assert unguarded.shape == (0, HIDDEN)

    with pytest.raises(AssertionError, match="empty grid side"):
        with validating(op):
            op(table, t, h, w, NUM_GRID, MERGE, torch.bfloat16)


@pytest.mark.parametrize("hw", [(6, 9), (5, 8)], ids=["w_odd", "h_odd"])
def test_guard_refuses_a_side_the_merge_does_not_divide(hw) -> None:
    """The wrapper refuses this only with a bare `assert`, which `python -O`
    strips; the kernel then reads outside the grid (measured, recorded in the
    entry's note, not driven here). The guard's own refusal comes first."""
    table = _table(34)
    h, w = hw
    with pytest.raises(AssertionError, match="must be divisible by m_size"):
        op(table, 1, h, w, NUM_GRID, MERGE, torch.bfloat16)

    with pytest.raises(AssertionError, match="not whole m_size=2 merge blocks"):
        with validating(op):
            op(table, 1, h, w, NUM_GRID, MERGE, torch.bfloat16)
