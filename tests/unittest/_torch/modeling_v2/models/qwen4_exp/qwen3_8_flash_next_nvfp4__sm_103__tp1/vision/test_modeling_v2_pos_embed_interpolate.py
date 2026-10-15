# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp pos_embed_interpolate entry.

`op.reference` is the explicit four-corner sum in fp64 -- exact, but sharing
the entry's sample positions, so it gates numerics. `_ref_resize` here runs
torch's own `F.interpolate` in fp64 and shares no arithmetic with the entry, so
it gates the structure: which axis is which, how the table is read, and the
spatial-merge row order. It sits up to 8.6e-6 from the exact value (its sample
coordinates are float), which is why fp32 cells gate on `op.reference`.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.vision.pos_embed_interpolate import (  # noqa: E501
    pos_embed_interpolate as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "pos_embed_interpolate requires a CUDA device"

NUM_GRID = 48
NUM_POS = NUM_GRID * NUM_GRID
HIDDEN = 1152
MERGE = 2


def _table(rows=NUM_POS, hidden=HIDDEN, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, hidden, generator=g, device="cuda").to(dtype)


def _merge_order(x, height, width, merge):
    hidden = x.shape[-1]
    return (
        x.reshape(height // merge, merge, width // merge, merge, hidden)
        .permute(0, 2, 1, 3, 4)
        .reshape(height * width, hidden)
    )


def _ref_resize(table, grid_thw, merge, mode="bilinear", align_corners=True):
    """Structural reference via torch's resize kernel, in fp64."""
    num_grid = int(round(table.shape[0] ** 0.5))
    hidden = table.shape[1]
    src = table.double().view(num_grid, num_grid, hidden).permute(2, 0, 1).unsqueeze(0).contiguous()
    outs = []
    for frames, height, width in grid_thw:
        kwargs = {} if mode == "nearest" else {"align_corners": align_corners}
        resized = F.interpolate(src, size=(height, width), mode=mode, **kwargs)
        rows = _merge_order(resized[0].permute(1, 2, 0), height, width, merge)
        outs.append(rows.repeat(frames, 1) if frames > 1 else rows)
    return torch.cat(outs) if len(outs) > 1 else outs[0]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    table = _table(spec["rows"], spec["hidden"], spec["dtype"], seed=zlib.crc32(cell.why.encode()))
    grid = [list(g) for g in spec["grid"]]
    before = table.clone()
    out = op(table, grid, spec["merge"])
    assert out.shape == (sum(t * h * w for t, h, w in grid), spec["hidden"])
    assert out.dtype == table.dtype
    assert torch.equal(table, before) and out.data_ptr() != table.data_ptr()
    torch.testing.assert_close(out, op.reference(table, grid, spec["merge"]))
    if spec["dtype"] is not torch.float32:
        # The structural gate, from a reference that shares no arithmetic.
        torch.testing.assert_close(out, _ref_resize(table, grid, spec["merge"]).to(out.dtype))


def test_exact_grid_is_the_table_itself_in_merge_order() -> None:
    """At h = w = N every sample lands on a cell; asserted bitwise. This is what
    pins the corner-aligned convention -- a cell-centre sampler lacks it."""
    table = _table(seed=1)
    out = op(table, [(1, NUM_GRID, NUM_GRID)], MERGE)
    direct = _merge_order(table.view(NUM_GRID, NUM_GRID, HIDDEN), NUM_GRID, NUM_GRID, MERGE)
    assert torch.equal(out, direct)


def test_the_two_grid_axes_are_not_interchangeable() -> None:
    """The bug a square test image hides: (1, 4, 8) and (1, 8, 4) differ by ~4.7."""
    table = _table(seed=2)
    tall = op(table, [(1, 8, 4)], MERGE)
    wide = op(table, [(1, 4, 8)], MERGE)
    assert (tall.float() - wide.float()).abs().max().item() > 1.0


def test_discrimination_against_the_wrong_operators() -> None:
    """Six plausible mis-implementations at [2304, 1152] bf16, grid (1, 62, 90).

    Measured max deviation from the entry: correct 0.0147; axes transposed
    6.31; raster order 6.06; intra-block transpose 4.63; align_corners=False
    1.38; column-major table read 6.31; nearest neighbour 5.64.
    """
    table = _table(seed=3)
    grid = [(1, 62, 90)]
    height, width = 62, 90
    out = op(table, grid, MERGE).double()
    correct_dev = (out - _ref_resize(table, grid, MERGE)).abs().max().item()

    src = table.double().view(NUM_GRID, NUM_GRID, HIDDEN).permute(2, 0, 1).unsqueeze(0).contiguous()
    raster = F.interpolate(src, size=(height, width), mode="bilinear", align_corners=True)
    raster = raster[0].permute(1, 2, 0)
    transposed = F.interpolate(src, size=(width, height), mode="bilinear", align_corners=True)
    transposed = transposed[0].permute(1, 2, 0).transpose(0, 1).contiguous()
    col_major = table.double().view(NUM_GRID, NUM_GRID, HIDDEN).permute(2, 1, 0).unsqueeze(0)
    col_major = F.interpolate(
        col_major.contiguous(), size=(height, width), mode="bilinear", align_corners=True
    )[0].permute(1, 2, 0)
    intra = raster.reshape(height // MERGE, MERGE, width // MERGE, MERGE, HIDDEN)
    intra = intra.permute(0, 2, 3, 1, 4).reshape(height * width, HIDDEN)

    wrong = {
        "transposed_axes": _merge_order(transposed, height, width, MERGE),
        "raster_order": raster.reshape(height * width, HIDDEN),
        "intra_block_transpose": intra,
        "align_corners_false": _ref_resize(table, grid, MERGE, align_corners=False),
        "column_major_table": _merge_order(col_major, height, width, MERGE),
        "nearest_neighbour": _ref_resize(table, grid, MERGE, mode="nearest"),
    }
    print(f"\n  correct {correct_dev:.6f}")
    for name, variant in wrong.items():
        dev = (out - variant).abs().max().item()
        print(f"  {name:24s} {dev:.6f}")
        assert dev > 20 * correct_dev, (name, dev, correct_dev)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out.to(table.dtype), variant.to(table.dtype))


def test_correctly_rounded_and_not_a_bf16_accumulate_chain() -> None:
    """fp32 accumulation, one rounding: within one bf16 ulp of the fp64 value
    and nearer it than the bf16 multiply-then-add chain a literal HF
    transcription computes."""
    table = _table(seed=4)
    for grid in ([(1, 62, 90)], [(1, 64, 64)]):
        _, height, width = grid[0]
        out = op(table, grid, MERGE)
        exact = op.reference(table.double(), grid, MERGE)

        h_pos = torch.linspace(0, NUM_GRID - 1, height, device="cuda")
        w_pos = torch.linspace(0, NUM_GRID - 1, width, device="cuda")
        h_lo, w_lo = h_pos.to(torch.int64), w_pos.to(torch.int64)
        h_hi = (h_lo + 1).clamp(max=NUM_GRID - 1)
        w_hi = (w_lo + 1).clamp(max=NUM_GRID - 1)
        d_h, d_w = (h_pos - h_lo).unsqueeze(1), (w_pos - w_lo).unsqueeze(0)
        grid_bf = table.view(NUM_GRID, NUM_GRID, HIDDEN)
        chain = (
            grid_bf[h_lo][:, w_lo] * ((1 - d_h) * (1 - d_w)).bfloat16()[..., None]
            + grid_bf[h_lo][:, w_hi] * ((1 - d_h) * d_w).bfloat16()[..., None]
            + grid_bf[h_hi][:, w_lo] * (d_h * (1 - d_w)).bfloat16()[..., None]
            + grid_bf[h_hi][:, w_hi] * (d_h * d_w).bfloat16()[..., None]
        )
        chain = _merge_order(chain, height, width, MERGE)

        entry_dev = (out.double() - exact).abs().max().item()
        chain_dev = (chain.double() - exact).abs().max().item()
        print(f"\n  {grid}: entry {entry_dev:.6f}, bf16 chain {chain_dev:.6f}")
        assert entry_dev < chain_dev, (grid, entry_dev, chain_dev)
        assert entry_dev <= 2.0**-8 * exact.abs().max().item(), (grid, entry_dev)


def test_a_batch_is_the_concatenation_of_its_images() -> None:
    table = _table(seed=7)
    grid = [(1, 4, 8), (2, 6, 4), (1, 48, 48), (1, 62, 90)]
    batched = op(table, grid, MERGE)
    one_by_one = torch.cat([op(table, [g], MERGE) for g in grid])
    assert torch.equal(batched, one_by_one)


def test_video_frames_repeat_the_spatial_embedding() -> None:
    table = _table(seed=8)
    frames, height, width = 3, 6, 10
    per_frame = op(table, [(frames, height, width)], MERGE).view(frames, height * width, HIDDEN)
    for f in range(1, frames):
        assert torch.equal(per_frame[f], per_frame[0]), f
    assert torch.equal(per_frame[0], op(table, [(1, height, width)], MERGE))


def test_guard_refuses_a_non_square_table() -> None:
    """The hazard, unguarded: 2300 rows -> isqrt 47, every index in range, a
    correctly shaped result off the wrong grid."""
    table = _table(rows=2300, hidden=64, seed=10)
    out = op(table, [(1, 4, 4)], MERGE)
    assert out.shape == (16, 64)
    assert torch.equal(out, op(table[: 47 * 47], [(1, 4, 4)], MERGE)), "read as a 47x47 grid"

    with pytest.raises(AssertionError, match="must hold a square grid"):
        with validating(op):
            op(table, [(1, 4, 4)], MERGE)


def test_grid_not_divisible_by_the_merge_size_raises() -> None:
    """Loud from torch, so not guarded."""
    table = _table(hidden=64, seed=11)
    with pytest.raises(RuntimeError, match="is invalid for input of size"):
        op(table, [(1, 5, 8)], MERGE)
    with pytest.raises(RuntimeError, match="is invalid for input of size"):
        op(table, [(1, 8, 5)], MERGE)
    with pytest.raises(RuntimeError, match="is invalid for input of size"):
        op(table, [(1, 2, 2)], 4)


def test_non_2d_table_raises() -> None:
    table = torch.randn(NUM_POS, 4, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="weight has to be a 2D Tensor"):
        op(table, [(1, 4, 4)], MERGE)


def test_cuda_graph_capture_and_replay() -> None:
    """The grid is frozen at capture; the table's contents are not."""
    table = _table(seed=12)
    grid = [(1, 8, 12)]

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(table, grid, MERGE)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(table, grid, MERGE)

    table.copy_(_table(seed=13))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, op.reference(table, grid, MERGE))
