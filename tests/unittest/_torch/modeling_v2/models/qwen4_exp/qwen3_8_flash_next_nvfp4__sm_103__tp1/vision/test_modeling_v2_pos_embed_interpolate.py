# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp pos_embed_interpolate entry.

`op.reference` is the explicit four-corner sum in fp64 -- exact, but sharing
the entry's sample positions, so it gates numerics. `_ref_resize` here runs
torch's own `F.interpolate` in fp64 and shares no arithmetic with the entry, so
it gates the structure: which axis is which, how the table is read, and the
spatial-merge row order. It sits up to 8.6e-6 from the exact value (its sample
coordinates are float): far inside a bf16 cell's band, though not an fp32 one's.
Beyond the cells: the square-table guard and graph capture.
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


def _ref_resize(table, grid_thw, merge):
    """Structural reference via torch's resize kernel, in fp64."""
    num_grid = int(round(table.shape[0] ** 0.5))
    hidden = table.shape[1]
    src = table.double().view(num_grid, num_grid, hidden).permute(2, 0, 1).unsqueeze(0).contiguous()
    outs = []
    for frames, height, width in grid_thw:
        resized = F.interpolate(src, size=(height, width), mode="bilinear", align_corners=True)
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
    # The structural gate, from a reference that shares no arithmetic.
    torch.testing.assert_close(out, _ref_resize(table, grid, spec["merge"]).to(out.dtype))


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
