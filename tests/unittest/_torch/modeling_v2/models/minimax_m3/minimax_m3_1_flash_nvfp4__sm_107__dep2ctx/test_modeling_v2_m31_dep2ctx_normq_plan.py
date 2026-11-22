# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_NORMQ ring boundary geometry (ctx_boundary_plan.py), CPU only, no torch.

The plan mirrors ctxBoundaryNormQuantSm107.cuh's launch math: every row of the padded scale tiles is visited by exactly
one quad slot, the producer loads the consumer's rows in the consumer's order, the quad's 16-byte scale-group stores put
every byte where get_sf_out_offset_128x4 (the served layout) puts it, once; CBNQ_P2's chunk map covers each row once
and its quads cover the rows with a doubled grid. The module is loaded from its file so the test also runs where torch
is not installed.
"""

import importlib.util
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[7]
_PLAN = (
    _REPO / "tensorrt_llm/_torch/modeling_v2/models/minimax_m3/"
    "minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/ctx_boundary_plan.py"
)
_spec = importlib.util.spec_from_file_location("ctx_boundary_plan", _PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

pytestmark = pytest.mark.cpu_only

H = 6144


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 4097, 16384])
def test_quads_cover_the_padded_rows_once(rows):
    padded = (rows + 127) // 128 * 128
    seen = [plan.quad_row(q, j) for q in range(plan.num_quads(rows)) for j in range(plan.QUAD_ROWS)]
    assert sorted(seen) == list(range(padded))


@pytest.mark.parametrize("rows", [1, 1000, 16384])
@pytest.mark.parametrize("grid_size", [1, 148])
def test_the_producer_loads_the_consumer_rows_in_order(rows, grid_size):
    g = min(grid_size, plan.num_quads(rows))
    every = []
    for cta in range(g):
        consumed = [r for _, _, r in plan.cta_rows(cta, g, rows) if r < rows]
        assert plan.producer_rows(cta, g, rows) == consumed
        every += consumed
    assert sorted(every) == list(range(rows))


@pytest.mark.parametrize("hidden", [256, 6144])
@pytest.mark.parametrize("rows", [1, 128, 300])
def test_quad_scale_stores_match_the_served_layout(hidden, rows):
    ks = range(hidden // 32)
    for q in range(plan.num_quads(rows)):
        for j in range(plan.QUAD_ROWS):
            row = plan.quad_row(q, j)
            for k in ks:
                assert plan.quad_sf_offset(q, j, k, hidden) == plan.served_sf_offset(
                    row, k, rows, hidden
                )


def test_every_scale_group_is_stored_once():
    rows, hidden = 300, 1024
    groups = {}
    for q in range(plan.num_quads(rows)):
        for kt in range(hidden // 128):
            base = plan.quad_sf_offset(q, 0, kt * 4, hidden)
            assert base % 16 == 0
            groups[base] = groups.get(base, 0) + 1
    total = (rows + 127) // 128 * 128 * (hidden // 32)
    assert sorted(groups) == list(range(0, total, 16)) and set(groups.values()) == {1}


# ---- CBNQ_P2: two chunks per thread, two CTAs per SM -----------------------------------------------------------------
def test_p2_chunks_cover_the_row_once():
    for hidden in (512, 4096, 6144):
        block = plan.p2_block(hidden)
        chunks = [plan.p2_chunk(t, h, hidden) for t in range(block) for h in range(plan.P2_CHUNKS)]
        assert sorted(chunks) == list(range(hidden // plan.PACKED))
    assert plan.p2_block(H) == 384 and plan.p2_virtual_warps(H) == 24 == H // 8 // 32


@pytest.mark.parametrize("rows", [1, 1000, 1160, 8192, 16317, 16384])
def test_p2_quads_cover_rows_with_double_grid(rows):
    g = plan.grid(rows, 2, 212)
    assert g == min(plan.num_quads(rows), 424)
    every, slots = [], []
    for cta in range(g):
        visits = plan.cta_rows(cta, g, rows)
        slots += [r for _, _, r in visits]
        consumed = [r for _, _, r in visits if r < rows]
        assert plan.producer_rows(cta, g, rows) == consumed
        every += consumed
    padded = (rows + 127) // 128 * 128
    assert sorted(slots) == list(range(padded)) and sorted(every) == list(range(rows))
