# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_swa_slots entry: per-step SWA window slots and visible lengths of every layer.

Integer index arithmetic only, so every cell gates the kernel against the plain-torch reference bit for bit.
"""

import pytest
import torch
from _v41_util import DEV, IS_SM103, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_swa_slots import (  # noqa: E501
    csa2_swa_slots,
    csa2_swa_table_descriptors,
)

pytestmark = pytest.mark.skipif(
    not IS_SM103, reason="the slot kernel and its CUDA-graph replay run on SM103 only"
)


def _ids(c):
    return [x.why[:48] for x in c]


def _inputs(spec, gen):
    layers, swa_layers, nreq, rows, window = (
        spec[k] for k in ("layers", "swa_layers", "nreq", "rows", "window")
    )
    start = spec["start"]
    if start == "random":
        start = torch.randint(0, 3000, (nreq,), generator=gen, device=DEV).tolist()
    blocks = 40
    tables = torch.stack(
        [
            torch.randperm(nreq * blocks + 7, generator=gen, device=DEV)[: nreq * blocks]
            .view(nreq, blocks)
            .int()
            for _ in range(swa_layers)
        ]
    )
    # the live pages of a request are the ones its window touches; the rest is unmapped (-1)
    live = torch.zeros(nreq, blocks, dtype=torch.bool, device=DEV)
    pos = []
    for r in range(nreq):
        for j in range(rows):
            pos.append(start[r] + j)
        live[r, max(0, (start[r] - 127) // 128) : (start[r] + rows) // 128 + 1] = True
    tables = torch.where(live[None], tables, torch.full_like(tables, -1))
    if spec["start"] == "random":  # also an unmapped page inside some windows
        tables[:, ::7, 0] = -1
    pos = torch.tensor(pos, dtype=torch.int32, device=DEV)
    rid = torch.arange(nreq, device=DEV).repeat_interleave(rows)
    ratios = torch.tensor(
        [0, 0] + [2] * 6 + [1] * (layers - 8) if layers >= 8 else [2] * layers,
        dtype=torch.int32,
        device=DEV,
    )[:layers]
    floors = torch.zeros(nreq, dtype=torch.int64, device=DEV)
    return pos, rid, tables.contiguous(), ratios, floors, window


@pytest.mark.parametrize("cell", csa2_swa_slots.CELLS, ids=_ids(csa2_swa_slots.CELLS))
def test_swa_slots_cells(cell, certifying):  # noqa: F811
    certifying(csa2_swa_slots)
    pos, rid, tables, ratios, floors, window = _inputs(cell.spec, gen_for(cell))
    desc = csa2_swa_table_descriptors(tables)
    args = (pos, rid, tables, desc, ratios, floors, window, 128)
    with validating(csa2_swa_slots):
        got = csa2_swa_slots(*args)
    ref = csa2_swa_slots.reference(*args)
    for name, g, r in zip(("reads", "writes", "visible"), got, ref):
        assert g.shape == r.shape and g.dtype == torch.int64, name
        assert torch.equal(g, r), name
    reads = got[0]
    assert (reads == -1).any() and (reads >= 0).any(), (
        "the cell must exercise mapped and unmapped rows"
    )


def test_swa_slots_worked_example():
    tables = torch.tensor([[[3, 1, -1]]], dtype=torch.int32, device=DEV)  # one layer, one request
    desc = csa2_swa_table_descriptors(tables)
    pos = torch.tensor([130, 5], dtype=torch.int32, device=DEV)
    rid = torch.zeros(2, dtype=torch.int64, device=DEV)
    ratios = torch.tensor(
        [2, 0], dtype=torch.int32, device=DEV
    )  # layer 1 owns no SWA pool: visible only
    floors = torch.zeros(1, dtype=torch.int64, device=DEV)
    with validating(csa2_swa_slots):
        reads, writes, visible = csa2_swa_slots(pos, rid, tables, desc, ratios, floors, 4, 128)
    # position 130 sits in block 1 (page 1): row 1 * 128 + 2; its window is 127..130
    assert reads[0, 0].tolist() == [3 * 128 + 127, 128 + 0, 128 + 1, 128 + 2]
    assert writes[0].tolist() == [128 + 2, 3 * 128 + 5]
    assert reads[0, 1].tolist() == [3 * 128 + 2, 3 * 128 + 3, 3 * 128 + 4, 3 * 128 + 5]
    assert visible.tolist() == [[65, 3], [0, 0]]


def test_cuda_graph_replay_follows_device_data():
    """Positions are device data: a replay with new positions must give what an eager call gives."""
    spec = dict(layers=8, swa_layers=6, nreq=4, rows=6, start=(5, 127, 300, 900), window=128)
    pos, rid, tables, ratios, floors, window = _inputs(
        spec, torch.Generator(device=DEV).manual_seed(1)
    )
    desc = csa2_swa_table_descriptors(tables)
    run = lambda p: csa2_swa_slots(p, rid, tables, desc, ratios, floors, window, 128)  # noqa: E731
    run(pos)  # eager warm-up (kernel JIT)
    torch.cuda.synchronize()
    static_pos = pos.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run(static_pos)
    static_pos.copy_(pos + 3)
    g.replay()
    torch.cuda.synchronize()
    ref = run(pos + 3)
    assert all(torch.equal(a, b) for a, b in zip(out, ref))
