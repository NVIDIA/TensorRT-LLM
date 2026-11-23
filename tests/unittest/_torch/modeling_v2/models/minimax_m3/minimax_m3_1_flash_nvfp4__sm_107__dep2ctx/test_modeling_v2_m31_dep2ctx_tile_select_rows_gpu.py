# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_TSEL_ROWS on sm_107: ``minimax_m31_select_prefill_tile_rows`` on a row-major view (unit block
stride, the Q8KV8 scorer's [T * H, pitch] buffer as [H, pages, T]) writes the table ``minimax_m31_select_prefill_tile``
writes on the same values head-major, BIT FOR BIT, and the table ``minimax_m3_select_blocks`` writes on the view itself.

Adversarial values (ties, +-0, both NaN signs, +-inf, all-equal rows, short rows) over block counts across the narrow /
tile / wide boundaries (128 / 2600) up to the 5856-block bound, 1 / 4 KV heads, a row count off the 8-row tile,
pitches 0 and 1 mod 4 and a 4-byte base offset (the scalar staging path), random per-row counts (rows of one tile end
apart). The built-in select ranks negative NaN above finite scores from 129 blocks (see the tp2ctx tile select test's
strict xfail), so that comparison leaves negative NaN out there. Skipped off sm_107.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx.m31_switches import (  # noqa: E501
    m31_disabled,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.tile_select import (
    TileSelect,
    TileSelectSettings,
)
from tensorrt_llm._utils import get_sm_version

# The dep2ctx target's tile select (its modeling module builds the same one).
_TSEL_ON = not m31_disabled("DEP2CTX_TSEL")
tile_select = TileSelect(
    TileSelectSettings(
        target="dep2ctx",
        switch="DEP2CTX_TSEL",
        enabled=_TSEL_ON,
        rows_enabled=_TSEL_ON and not m31_disabled("DEP2CTX_TSEL_ROWS"),
        narrow=not m31_disabled("TSEL_NARROW"),
    )
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)
POS_NAN, NEG_NAN = 0x7FC00000, -0x00400000


def _values(kind, heads, blocks, rows, seed, neg_nan):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (heads, blocks, rows)
    if kind == "random":
        return torch.randn(shape, generator=g, device="cuda")
    if kind == "ties":
        return torch.randint(0, 8, shape, generator=g, device="cuda").float()
    if kind == "all_equal":
        return torch.full(shape, 0.5, device="cuda")
    if kind == "specials":
        s = torch.randint(-3, 4, shape, generator=g, device="cuda").float()
        bits = s.view(torch.int32)
        pick = torch.randint(0, 12, shape, generator=g, device="cuda")
        special = torch.tensor(
            [0x7F800000, -0x00800000, POS_NAN, NEG_NAN if neg_nan else POS_NAN, 0, -0x80000000],
            dtype=torch.int32, device="cuda")  # fmt: skip
        return torch.where(pick < 6, special[pick.clamp(max=5)], bits).view(torch.float32)
    raise ValueError(kind)


def _row_major(values, pitch_mod, offset):
    """``values`` [H, B, T] copied into a [T * H, pitch] buffer (row t * H + h), pitch % 4 == pitch_mod, starting
    ``offset`` floats into its allocation; returned as the [H, B, T] view the scorer hands the select."""
    heads, blocks, rows = values.shape
    pitch = (blocks + 3) // 4 * 4 + pitch_mod
    flat = torch.full((offset + rows * heads * pitch,), float("nan"), device="cuda")
    buf = flat[offset:].view(rows * heads, pitch)
    view = buf.view(rows, heads, pitch)[:, :, :blocks].permute(1, 2, 0)
    view.copy_(values)
    assert view.stride(1) == 1 and torch.equal(
        view.nan_to_num(nan=-3.0), values.nan_to_num(nan=-3.0)
    )
    return view


def _tables(values, view, nvb, init_blocks, local_blocks):
    heads, blocks, rows = values.shape
    tile = torch.empty((heads, rows, 16), dtype=torch.int32, device="cuda").permute(1, 0, 2)
    rows_out = torch.empty_like(tile.permute(1, 0, 2)).permute(1, 0, 2)
    family_ops().minimax_m31_select_prefill_tile(
        values.contiguous(), nvb, tile, init_blocks, local_blocks
    )
    family_ops().minimax_m31_select_prefill_tile_rows(
        view, nvb, rows_out, init_blocks, local_blocks
    )
    # The prefill scorer's selector routes the view to the rows op; the general selector hook leaves it alone.
    routed = tile_select.select_rows(view, nvb, init_blocks, local_blocks)
    assert tile_select.select(view, nvb, init_blocks, local_blocks) is None
    builtin = torch.ops.trtllm.minimax_m3_select_blocks(
        view, nvb, 16, init_blocks, local_blocks, True
    )
    return tile, rows_out, routed, builtin


def _diff(a, b):
    d = (a != b).any(dim=-1)
    return f"{int(d.sum())} rows differ; first {d.nonzero()[:4].tolist()}"


def _same_selection_shape(a, b, nvb):
    """Both tables pick the same number of blocks per (row, head), each pick is a block the row can see, and no block
    is picked twice."""
    limit = nvb.long().view(-1, 1, 1)
    for t in (a, b):
        valid = t >= 0
        if bool((valid & (t.long() >= limit)).any()):
            return False
        picked = torch.where(valid, t, torch.full_like(t, -1)).sort(dim=-1).values
        if bool(((picked[..., 1:] == picked[..., :-1]) & (picked[..., 1:] >= 0)).any()):
            return False
    return torch.equal((a >= 0).sum(dim=-1), (b >= 0).sum(dim=-1))


@pytest.mark.parametrize("blocks", [128, 129, 5856])
@pytest.mark.parametrize("rows", [1001])
@pytest.mark.parametrize("heads", [4])
@pytest.mark.parametrize("init_blocks,local_blocks", [(2, 3)])
@pytest.mark.parametrize("pitch_mod,offset", [(0, 0), (1, 1)])
def test_rows_op_equals_the_tile_op_bitwise(
    blocks, rows, heads, init_blocks, local_blocks, pitch_mod, offset
):
    g = torch.Generator(device="cuda").manual_seed(blocks * 31 + rows)
    # Random counts in [0, blocks + 3]: rows of one tile end apart, some past the table, some empty.
    nvb = torch.randint(0, blocks + 4, (rows,), generator=g, device="cuda", dtype=torch.int32)
    for kind in ("random", "ties", "all_equal", "specials"):
        neg_nan = blocks <= 128
        values = _values(kind, heads, blocks, rows, blocks + rows + heads, neg_nan)
        view = _row_major(values, pitch_mod, offset)
        tile, rows_out, routed, builtin = _tables(values, view, nvb, init_blocks, local_blocks)
        assert torch.equal(rows_out, tile), f"{kind}: {_diff(rows_out, tile)}"
        assert routed is not None and torch.equal(routed, tile), kind
        if kind == "all_equal":
            # Every score ties: the tile op and the built-in select keep different, equally valid top-16 sets past
            # ~2.6K blocks (64 rows at 2599 / 2600 / 5856 blocks on SM107). The rows op's claim is equality with the
            # tile op (above); against the built-in only the selection's shape holds.
            assert _same_selection_shape(rows_out, builtin, nvb), kind
        else:
            assert torch.equal(rows_out, builtin), f"{kind}: {_diff(rows_out, builtin)}"


@pytest.mark.parametrize("ctx_end,rows", [(16384, 16384), (165000, 1160), (65536, 4096)])
def test_served_shapes_with_causal_counts(ctx_end, rows):
    """Served-like rows: one request's last ``rows`` tokens at context ``ctx_end`` (counts climb by one block every 128
    rows), 4 KV heads."""
    blocks = (ctx_end + 127) // 128
    pos = torch.arange(ctx_end - rows, ctx_end, device="cuda")
    nvb = ((pos + 128) // 128).to(torch.int32)
    values = _values("random", 4, blocks, rows, ctx_end, True)
    view = _row_major(values, 0, 0)
    tile, rows_out, routed, builtin = _tables(values, view, nvb, 0, 1)
    assert (
        torch.equal(rows_out, tile) and torch.equal(routed, tile) and torch.equal(rows_out, builtin)
    )
